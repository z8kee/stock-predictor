const chartOptions = { 
    layout: { textColor: '#d1d4dc', background: { type: 'solid', color: '#1e1e1e' } },
    grid: { vertLines: { color: '#2B2B43' }, horzLines: { color: '#2B2B43' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal }
};

const chart = LightweightCharts.createChart(document.getElementById('tvchart'), chartOptions);

const candleSeries = chart.addCandlestickSeries({
    upColor: '#2aa626', downColor: '#ef5350', borderVisible: false, wickUpColor: '#2aa626', wickDownColor: '#ef5350'
});

async function fetchAndRenderChart(ticker, interval) {
    console.log(`Fetching data for ${ticker} at ${interval}...`);
    const response = await fetch(`/api/history/${ticker}/${interval}`);
    const data = await response.json();
    
    candleSeries.setData(data);
    chart.timeScale().fitContent(); 
}

const assetSelect = document.getElementById('Select-Stock');
const intervalSelect = document.getElementById('Select-Timeframe');

function updateChart() {
    const currentTicker = assetSelect.value;
    const currentInterval = intervalSelect.value;
    fetchAndRenderChart(currentTicker, currentInterval);
}

assetSelect.addEventListener('change', updateChart);
intervalSelect.addEventListener('change', updateChart);

updateChart();