const chartOptions = { 
            layout: { textColor: '#d1d4dc', background: { type: 'solid', color: '#554f4f' } },
            grid: { vertLines: { color: '#2B2B43' }, horzLines: { color: '#2B2B43' } },
            crosshair: { mode: LightweightCharts.CrosshairMode.Normal }
        };
        const chart = LightweightCharts.createChart(document.getElementById('tvchart'), chartOptions);
        
        const candleSeries = chart.addCandlestickSeries({
            upColor: '#26a69a', downColor: '#ef5350', borderVisible: false, wickUpColor: '#26a69a', wickDownColor: '#ef5350'
        });

        // 3. Function to pull data from our Python Backend
        async function fetchAndRenderChart(ticker, interval) {
            console.log(`Fetching data for ${ticker}...`);
            const response = await fetch(`/api/history/${ticker}/${interval}`);
            const data = await response.json();
            
            // Feed the data into the chart
            candleSeries.setData(data);
            chart.timeScale().fitContent(); // Auto-zoom to fit
        }

        // Grab both dropdown elements
        const assetSelect = document.getElementById('Select Stock');
        const intervalSelect = document.getElementById('Select Timeframe');

        // Helper function that reads both boxes at the exact same time
        function updateChart() {
            const currentTicker = assetSelect.value;
            const currentInterval = intervalSelect.value;
            fetchAndRenderChart(currentTicker, currentInterval);
        }

        // 4. Add Event Listeners to BOTH Dropdown Menus
        // Now, if you change either box, it triggers the updateChart function
        assetSelect.addEventListener('change', updateChart);
        intervalSelect.addEventListener('change', updateChart);

        // 5. Load the default chart using whatever is currently selected in the HTML
        fetchAndRenderChart(assetSelect.value, intervalSelect.value);

        // 5. Load Gold by default when the page first opens
        fetchAndRenderChart('GC=F', "5m");