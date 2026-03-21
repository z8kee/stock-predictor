const assetSelect = document.getElementById('Select-Stock');
const intervalSelect = document.getElementById('Select-Timeframe');
const loadingBar = document.getElementById('loading-bar');
const newsBtn = document.querySelector('.News-btn');
const predictBtn = document.querySelector('.Predict-btn');

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
    height: 500,
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
        data = data.filter((v, i, a) => a.findIndex(t => t.time === v.time) === i);
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
            // we slice the last 5 candles just in case of slight network lag
            const latestCandles = data.slice(-5);
            for (const candle of latestCandles) {
                candleSeries.update(candle);
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

    const currentTicker = assetSelect.value;
    const currentInterval = intervalSelect.value;

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

function queueNextUpdate() {
    updateTimer = setTimeout(async () => {
        const currentTicker = assetSelect.value;
        const currentInterval = intervalSelect.value;

        await fetchAndRenderChart(currentTicker, currentInterval, false);

        queueNextUpdate();
    }, 1500);
}

async function fetchAndRenderNews() {
    const ticker = assetSelect.value;
 
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
        if (!newsPanel) {
            newsPanel = document.createElement('div');
            newsPanel.id = 'news-panel';
            newsPanel.style.cssText = 'width: 80%; margin: 20px auto; text-align: left; background: #1e1e1e; padding: 20px; border-radius: 8px; border: 1px solid #333;';
            document.getElementById('tvchart').insertAdjacentElement('afterend', newsPanel);
        }
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
    const ticker = assetSelect.value;
    const interval = intervalSelect.value;

    try {
        const response = await fetch(`/api/predict/${ticker}/${interval}`);
        const result = await response.json();

        if (result.error) {
            console.error("Server returned an error:", result.error);
            return;
        }

        updatePredictionPanel(result);

        // only draws a chart marker if the signal is a BUY or SELL
        if (result.signal !== 'HOLD') {
            const allData = candleSeries.data();
            if (!allData || allData.length === 0) return;
            const lastCandle = allData[allData.length - 1];

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

function updatePredictionPanel(result) {
    const panel = document.getElementById('prediction-panel');
    const interval = intervalSelect.value;
    const black_swan_timeframes = {'1m': 1.0546, '5m': 2.5955, '15m': 2.7866, '1h': 2.4900, '1d': 0.8567}
    panel.style.display = 'block'; // unhide the panel

    const signalColor = result.signal === 'BUY' ? '#26a69a' : result.signal === 'SELL' ? '#ef5350' : '#888';
    let anomalyWarning = '';
    if (result.anomaly > black_swan_timeframes[interval]) {
        anomalyWarning = `
            <div style="margin-top: 12px; padding: 8px; background-color: rgba(255, 193, 7, 0.2); border: 1px solid #ffc107; border-radius: 4px; color: #ffc107; font-weight: bold; font-size: 0.9em; text-align: center;">
                ⚠️ BLACK SWAN WARNING: High market anomaly detected. Prediction confidence is compromised.
            </div>
        `;
    }
    
    panel.innerHTML = `
        <h2 style="margin-top:0">Latest Prediction</h2>
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

function startPredictions() {
    if (predictionTimer) {
        // Toggle off
        clearInterval(predictionTimer);
        predictionTimer = null;
        predictBtn.textContent = 'Start Prediction';
        predictBtn.style.borderColor = '#555';
        return;
    }

    // Toggle on
    predictBtn.textContent = 'Stop Prediction';
    predictBtn.style.borderColor = '#26a69a';
    fetchAndDisplayPrediction(); // run immediately
    predictionTimer = setInterval(fetchAndDisplayPrediction, 30000); // then every 30s
}

// dropdown listeners
assetSelect.addEventListener('change', handleManualUpdate);
intervalSelect.addEventListener('change', handleManualUpdate);
newsBtn.addEventListener('click', fetchAndRenderNews)

assetSelect.addEventListener('change', () => { signalMarkers = []; candleSeries.setMarkers([]); });
intervalSelect.addEventListener('change', () => { signalMarkers = []; candleSeries.setMarkers([]); });

predictBtn.addEventListener('click', startPredictions)

// run it
handleManualUpdate();

// made by zeeeeke gee pee tee
