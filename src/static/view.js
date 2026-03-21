const assetSelect = document.getElementById('Select-Stock');
const intervalSelect = document.getElementById('Select-Timeframe');
const loadingBar = document.getElementById('loading-bar');
const newsBtn = document.querySelector('.News-btn');
const predictBtn = document.querySelector('.Predict-btn');

// --- Chart Initialisation ---
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

// Resize chart when window resizes
window.addEventListener('resize', () => {
    chart.applyOptions({ width: document.getElementById('tvchart').clientWidth });
});

// Use a uniquely named variable - never shadow the built-in setTimeout!
let updateTimer = null;

async function fetchAndRenderChart(ticker, interval, isManualUpdate = true) {
    if (isManualUpdate) {
        showLoadingBar();
    }

    try {
        const response = await fetch(`/api/history/${ticker}/${interval}`);
        let data = await response.json();

        // 1. Force strict chronological order and remove duplicates
        data = data.filter((v, i, a) => a.findIndex(t => t.time === v.time) === i);
        data.sort((a, b) => a.time - b.time);

        if (isManualUpdate) {
            // INITIAL LOAD: Replace the whole chart and fit to screen
            candleSeries.setData(data);
            chart.timeScale().fitContent();
            
            // Re-apply markers if they exist
            if (signalMarkers.length > 0) {
                const validTimes = new Set(data.map(d => d.time));
                signalMarkers = signalMarkers.filter(m => validTimes.has(m.time)); 
                candleSeries.setMarkers(signalMarkers);
            }
        } else {
            // LIVE UPDATE: Just inject the latest data points smoothly
            // We slice the last 5 candles just in case of slight network lag
            const latestCandles = data.slice(-5);
            for (const candle of latestCandles) {
                candleSeries.update(candle);
            }
            // Notice we do NOT call setMarkers here! 
            // update() leaves your prediction arrows perfectly intact where they belong.
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
    // Stop any pending background update
    if (updateTimer) {
        clearTimeout(updateTimer);
        updateTimer = null;
    }

    const currentTicker = assetSelect.value;
    const currentInterval = intervalSelect.value;

    // Wait for full load before queuing the next update
    await fetchAndRenderChart(currentTicker, currentInterval, true);

    queueNextUpdate();
}

function showLoadingBar() {
    loadingBar.style.display = 'block';
    loadingBar.style.width = '0%';
    setTimeout(() => loadingBar.style.width = '70%', 50); // jump to 70% quickly
}

function hideLoadingBar() {
    loadingBar.style.width = '100%'; // finish to 100%
    setTimeout(() => {
        loadingBar.style.display = 'none';
        loadingBar.style.width = '0%'; // reset for next time
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

        // 1. ALWAYS update the prediction panel so you know the model actually ran
        updatePredictionPanel(result);

        // 2. ONLY draw a chart marker if the signal is a BUY or SELL
        if (result.signal !== 'HOLD') {
            const allData = candleSeries.data();
            if (!allData || allData.length === 0) return;
            const lastCandle = allData[allData.length - 1];

            // Safely grab the correct probability based on the signal
            const confidence = result.signal === 'BUY' ? result.probabilities.buy : result.probabilities.sell;

            const marker = {
                time: lastCandle.time,
                position: result.signal === 'BUY' ? 'belowBar' : 'aboveBar',
                color: result.signal === 'BUY' ? '#26a69a' : '#ef5350',
                shape: result.signal === 'BUY' ? 'arrowUp' : 'arrowDown',
                text: `${result.signal} ${(confidence * 100).toFixed(0)}%`
            };

            // Avoid duplicate markers at the same timestamp
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
    panel.style.display = 'block'; // Unhide the panel

    const signalColor = result.signal === 'BUY' ? '#26a69a' : result.signal === 'SELL' ? '#ef5350' : '#888';
    
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

// Dropdown listeners
assetSelect.addEventListener('change', handleManualUpdate);
intervalSelect.addEventListener('change', handleManualUpdate);
newsBtn.addEventListener('click', fetchAndRenderNews)

assetSelect.addEventListener('change', () => { signalMarkers = []; candleSeries.setMarkers([]); });
intervalSelect.addEventListener('change', () => { signalMarkers = []; candleSeries.setMarkers([]); });

predictBtn.addEventListener('click', startPredictions)

// Initial load
handleManualUpdate();