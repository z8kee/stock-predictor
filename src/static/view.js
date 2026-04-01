const stockSelect = document.getElementById('Select-Stock');
const timeframeSelect = document.getElementById('Select-Timeframe');
const loadingBar = document.getElementById('loading-bar');
const predictBtn = document.querySelector('.Predict-btn');
const newsBtn = document.querySelector('.News-btn');
const recBtn = document.querySelector('.Recommend-btn')

// initalise the chart
const chart = LightweightCharts.createChart(document.getElementById('tvchart'), {
    layout: {
        background: { color: '#121212' },
        textColor: '#d1d4dc',
    },
    grid: {
        vertLines: { color: '#636262' },
        horzLines: { color: '#636262' },
    },
    width: document.getElementById('tvchart').clientWidth,
    height: 700,
});

const candleSeries = chart.addCandlestickSeries({
    upColor: '#26a69a',
    downColor: '#ef5350',
    borderVisible: false,
    wickUpColor: '#26a69a',
    wickDownColor: '#ef5350',
});

// when window gets resized, we resize chart
window.addEventListener('resize', () => {
    chart.applyOptions({ width: document.getElementById('tvchart').clientWidth });
});

let updateTimer = null;

async function fetchAndRenderChart(ticker, interval, isManualUpdate = true) {
    if (isManualUpdate) {
        showLoadingBar();
    }

    try {
        const response = await fetch(`/api/history/${ticker}/${interval}`);
        let data = await response.json();

        // avoids duplicates and allows order
        const seenTimes = new Set();
        data = data.filter(d => {
            if (seenTimes.has(d.time)) return false;
            seenTimes.add(d.time);
            return true;
        });
        
        data.sort((a, b) => a.time - b.time);

        if (isManualUpdate) {
            candleSeries.setData(data);
            chart.timeScale().fitContent();
            
            if (signalMarkers.length > 0) {
                const validTimes = new Set(data.map(d => d.time));
                signalMarkers = signalMarkers.filter(m => validTimes.has(m.time)); 
                candleSeries.setMarkers(signalMarkers);
            }
        } else {
            // pass latest candle
            if (data.length > 0) {
                const latestCandle = data[data.length -1];
                candleSeries.update(latestCandle);
            }
        }

    } catch (error) {
        console.error("Error fetching data:", error);
    } finally {
        if (isManualUpdate) {
            hideLoadingBar();
        }
    }
}

async function handleManualUpdate() {
    // stop any pending background update
    if (updateTimer) {
        clearTimeout(updateTimer);
        updateTimer = null;
    }

    const currentTicker = stockSelect.value;
    const currentInterval = timeframeSelect.value;

    // wait for full load before queuing the next update
    await fetchAndRenderChart(currentTicker, currentInterval, true);

    queueNextUpdate();
}

function showLoadingBar() {
    loadingBar.style.display = 'block';
    loadingBar.style.width = '0%';
    setTimeout(() => loadingBar.style.width = '70%', 50);
}

function hideLoadingBar() {
    loadingBar.style.width = '100%'; 
    setTimeout(() => {
        loadingBar.style.display = 'none';
        loadingBar.style.width = '0%'; 
    }, 300);
}

function getDelay() {
    const update_delays = {
        '1m': 15000,       // 30 seconds
        '5m': 30000,      // 2.5 minutes
        '15m': 60000,     // 5 minutes
        '1h': 120000,      // 10 minutes
        '1d': 300000      // 30 minutes
    };
    return update_delays[timeframeSelect.value] || 15000;
}

function queueNextUpdate() {
    updateTimer = setTimeout(async () => {
        const currentTicker = stockSelect.value;
        const currentInterval = timeframeSelect.value;
    
        await fetchAndRenderChart(currentTicker, currentInterval, false);

        queueNextUpdate();
    }, getDelay());
}

async function fetchAndRenderNews() {
    const ticker = stockSelect.value;
 
    try {
        newsBtn.textContent = 'Loading...';
        newsBtn.disabled = true;
 
        const response = await fetch(`/api/news/${ticker}`);
        const newsData = await response.json();
 
        if (newsData.error) {
            alert('Could not fetch news: ' + newsData.error);
            return;
        }
 
        let html = "<ul style='list-style-type: none; padding: 0;'>";
        newsData.forEach(item => {
            const formattedScore = item.score.toFixed(2);
            html += `<li style="margin-bottom: 15px; border-bottom: 1px solid #333; padding-bottom: 10px;">
                <div style="margin-bottom: 5px;">
                    <strong>${item.sentiment}</strong>
                    <span style="color: #888; font-size: 0.9em;">(Net Score: ${formattedScore})</span>
                </div>
                <a href="${item.link}" target="_blank" style="color: #90caf9; text-decoration: none; font-size: 1.1em;">
                    ${item.title}
                </a>
            </li>`;
        });
        html += "</ul>";
 

        let newsPanel = document.getElementById('news-panel');
        newsPanel.style.display = 'block'; // Ensure it's visible
        newsPanel.innerHTML = `<h2 style="margin-top:0">Latest News</h2>` + html;
 
    } catch (error) {
        console.error("Error fetching news:", error);
        alert('Failed to fetch news.');
    } finally {
        newsBtn.textContent = 'View Current News for Stock';
        newsBtn.disabled = false;
    }
}

let signalMarkers = [];
let predictionTimer = null;

async function fetchAndDisplayPrediction() {
    const ticker = stockSelect.value;
    const interval = timeframeSelect.value;
    const recommended_thresholds = {'1m': {'BUY': 0.4496, 'SELL': 0.4404}, '5m': {'BUY': 0.4649, 'SELL': 0.5310},
                          '15m': {'BUY': 0.4448, 'SELL': 0.4240}, '1h': {'BUY': 0.3884, 'SELL': 0.4237},
                          '1d': {'BUY': 0.7132, 'SELL': 0.4974}
                          }
    try {
        const response = await fetch(`/api/predict/${ticker}/${interval}`);
        const result = await response.json();

        if (result.error) return;

        updatePredictionPanel(result);

        if (result.signal !== 'HOLD' && result.winning_prob >= recommended_thresholds[interval][result.signal]) {
            
            const allData = candleSeries.data();
            if (!allData || allData.length === 0) return;
            
            const lastCandle = allData[allData.length - 1];
            const currentIndex = allData.length - 1;

            const minGapCandles = 5; 
            
            if (signalMarkers.length > 0) {
                const lastMarker = signalMarkers[signalMarkers.length - 1];
                const lastMarkerIndex = allData.findIndex(d => d.time === lastMarker.time);

                if (lastMarkerIndex !== -1 && (currentIndex - lastMarkerIndex) < minGapCandles) {
                    return;
                }
            }
            const alertSound = new Audio('https://actions.google.com/sounds/v1/impacts/crash.ogg');
            alertSound.play().catch(e => console.log("Sound blocked by browser:", e));

            if (!allData || allData.length === 0) return;

            const confidence = result.signal === 'BUY' ? result.probabilities.buy : result.probabilities.sell;

            const marker = {
                time: lastCandle.time,
                position: result.signal === 'BUY' ? 'belowBar' : 'aboveBar',
                color: result.signal === 'BUY' ? '#26a69a' : '#ef5350',
                shape: result.signal === 'BUY' ? 'arrowUp' : 'arrowDown',
                text: `${result.signal} ${(confidence * 100).toFixed(0)}%`
            };

            signalMarkers = signalMarkers.filter(m => m.time !== lastCandle.time);
            signalMarkers.push(marker);
            signalMarkers.sort((a, b) => a.time - b.time);
            candleSeries.setMarkers(signalMarkers);
        }
    } catch (error) {
        console.error("Prediction error:", error);
    }
}

async function predictionLoop() {
    await fetchAndDisplayPrediction();
    if (predictionTimer !== null) {
        predictionTimer = setTimeout(predictionLoop, getDelay());
    }
}

function startPredictions() {
    if (predictionTimer) {
        // Toggle off
        clearTimeout(predictionTimer);
        predictionTimer = null;
        predictBtn.textContent = 'Start Prediction';
        predictBtn.style.borderColor = '#555';
        return;
    }

    // Toggle on
    predictBtn.textContent = 'Stop Prediction';
    predictBtn.style.borderColor = '#26a69a';
    predictionTimer = "starting"; // Placeholder to allow loop to run
    predictionLoop(); 
}

function updatePredictionPanel(result) {
    const panel = document.getElementById('prediction-panel');
    const interval = timeframeSelect.value;
    const stock = stockSelect.value;
    const black_swan_timeframes = {'1m': 1.7956, '5m': 2.5867, '15m': 2.7912, '1h': 2.5130, '1d': 0.8591}
    const recommended_thresholds = {'1m': {'BUY': 0.4496, 'SELL': 0.4404}, '5m': {'BUY': 0.4649, 'SELL': 0.5310},
                          '15m': {'BUY': 0.4448, 'SELL': 0.4240}, '1h': {'BUY': 0.3884, 'SELL': 0.4237},
                          '1d': {'BUY': 0.7132, 'SELL': 0.4974}
                          }
    panel.style.display = 'block'; // unhide the panel

    const signalColor = result.signal === 'BUY' && result.probabilities.buy >= recommended_thresholds[interval].BUY ? '#26a69a' : result.signal === 'SELL' && result.probabilities.sell >= recommended_thresholds[interval].SELL? '#ef5350' : '#888';
    let anomalyWarning = '';
    if (result.anomaly > black_swan_timeframes[interval]) {
        anomalyWarning = `
            <div style="margin-top: 12px; padding: 8px; background-color: rgba(255, 193, 7, 0.2); border: 1px solid #ffc107; border-radius: 4px; color: #ffc107; font-weight: bold; font-size: 0.9em; text-align: center;">
                High market anomaly detected. Prediction confidence is compromised.
            </div>
        `;
    }
    
    panel.innerHTML = `
        <h2 style="margin-top:0">Latest Prediction on ${interval} | ${stock}</h2>
        <div style="font-size: 1.4em; font-weight: bold; color: ${signalColor}">${result.signal}</div>
        <div style="margin-top: 10px; color: #aaa;">
            Buy: ${(result.probabilities.buy * 100).toFixed(1)}% &nbsp;|&nbsp;
            Hold: ${(result.probabilities.hold * 100).toFixed(1)}% &nbsp;|&nbsp;
            Sell: ${(result.probabilities.sell * 100).toFixed(1)}%
        </div>
        <div style="margin-top: 8px; color: #aaa; font-size: 0.9em;">
            Sentiment: ${result.sentiment.toFixed(3)} &nbsp;|&nbsp;
            Anomaly: ${result.anomaly.toFixed(4)}
        </div>
        ${anomalyWarning}
    `;
}

async function getAiRecommendations() {
    const ticker = stockSelect.value;
    const panel = document.getElementById('recommendation-panel');
    
    recBtn.textContent = 'Agent is Analyzing...';
    recBtn.disabled = true;
    
    try {
        const response = await fetch(`/api/recommendation/${ticker}/gpt-5.4-mini`);
        const data = await response.json();
        
        if (data.error) {
            alert('Agent Error: ' + data.error);
            return;
        }

        let recColor = '#888888';
        if (data.recommendation.includes('Buy')) recColor = '#26a69a';
        if (data.recommendation.includes('Sell')) recColor = '#ef5350';

        panel.style.display = 'block';
        panel.innerHTML = `
            <h3 style="margin-top:0; color: #ffffff;">What Today Looks Like on the ${timeframeSelect.value} chart</h3>
            <div style="font-size: 1.3em; font-weight: bold; color: ${recColor}; margin-bottom: 10px;">
                ${data.recommendation} Day
            </div>
            <div style="color: #aaa; line-height: 1.5; font-size: 0.95em;">
                ${data.description}
            </div>
        `;
    } catch (error) {
        console.error("Error fetching recommendation:", error);
    } finally {
        recBtn.textContent = 'Get AI Recommendation';
        recBtn.disabled = false;
    }
}

function restoreSelectionsFromStorage() {
    const savedStock = localStorage.getItem('selectedStock');
    const savedTimeframe = localStorage.getItem('selectedTimeframe');
    
    if (savedStock) stockSelect.value = savedStock;
    if (savedTimeframe) timeframeSelect.value = savedTimeframe;
}

function saveSelectionsToStorage() {
    localStorage.setItem('selectedStock', stockSelect.value);
    localStorage.setItem('selectedTimeframe', timeframeSelect.value);
}

stockSelect.addEventListener('change', () => {
    saveSelectionsToStorage();
    signalMarkers = [];
    candleSeries.setMarkers([]);
    handleManualUpdate();
});

timeframeSelect.addEventListener('change', () => {
    saveSelectionsToStorage();
    signalMarkers = [];
    candleSeries.setMarkers([]);
    handleManualUpdate();
});

predictBtn.addEventListener('click', startPredictions);
newsBtn.addEventListener('click', fetchAndRenderNews);
recBtn.addEventListener('click', getAiRecommendations);

// Restore and load
restoreSelectionsFromStorage();
handleManualUpdate();
