const assetSelect = document.getElementById('Select-Stock');
const intervalSelect = document.getElementById('Select-Timeframe');
const loadingBar = document.getElementById('loading-bar');
const newsBtn = document.querySelector('.News-btn');

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

// Use a uniquely named variable — never shadow the built-in setTimeout!
let updateTimer = null;

async function fetchAndRenderChart(ticker, interval, isManualUpdate = true) {
    if (isManualUpdate) {
        showLoadingBar();
    }

    try {
        console.log(`Fetching data for ${ticker} at ${interval}...`);
        const response = await fetch(`/api/history/${ticker}/${interval}`);
        const data = await response.json();

        candleSeries.setData(data);

        if (isManualUpdate) {
            chart.timeScale().fitContent();
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
    }, 2500);
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

// Dropdown listeners
assetSelect.addEventListener('change', handleManualUpdate);
intervalSelect.addEventListener('change', handleManualUpdate);
newsBtn.addEventListener('click', fetchAndRenderNews)

// Initial load
handleManualUpdate();