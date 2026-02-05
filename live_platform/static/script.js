// Global Chart Instance
let marketChart;
const maxDataPoints = 50;

// Initialize Chart
function initChart(historyData = []) {
    const ctx = document.getElementById('marketChart').getContext('2d');
    
    // Initial labels (placeholder if empty)
    const initialLabels = historyData.map((_, i) => i);
    
    marketChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: initialLabels,
            datasets: [{
                label: 'Gold Price (USD)',
                data: historyData,
                borderColor: '#fbbf24', // Yellow-400
                backgroundColor: 'rgba(251, 191, 36, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            interaction: {
                intersect: false,
                mode: 'index',
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: '#f8fafc',
                    bodyColor: '#fbbf24',
                    borderColor: '#334155',
                    borderWidth: 1,
                    displayColors: false,
                }
            },
            scales: {
                x: {
                    display: false,
                    grid: { display: false }
                },
                y: {
                    grid: {
                        color: 'rgba(51, 65, 85, 0.3)',
                        drawBorder: false,
                    },
                    ticks: {
                        color: '#94a3b8',
                        callback: function(value) { return '$' + value; }
                    }
                }
            }
        }
    });
}

// Connect to SSE Stream
function connectSSE() {
    const eventSource = new EventSource('/api/stream');
    const priceDisplay = document.getElementById('price-display');
    const priceChange = document.getElementById('price-change');
    const loadingScreen = document.getElementById('loading-screen');

    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        const price = data.price;
        const change = data.change_percent;

        // Update DOM
        priceDisplay.innerText = `$${price.toFixed(2)}`;
        
        const changeColor = change >= 0 ? 'text-green-400' : 'text-red-400';
        const sign = change >= 0 ? '+' : '';
        priceChange.innerHTML = `<span class="${changeColor} font-bold mr-2">${sign}${(change*100).toFixed(2)}%</span> <span class="text-slate-500 text-xs">Since last tick</span>`;

        // Update Chart
        if (marketChart) {
            const labels = marketChart.data.labels;
            const prices = marketChart.data.datasets[0].data;

            labels.push(data.timestamp);
            prices.push(price);

            if (labels.length > maxDataPoints) {
                labels.shift();
                prices.shift();
            }

            marketChart.update('none'); // Efficient update
        }

        // Hide loader on first data
        if (loadingScreen.style.opacity !== '0') {
            loadingScreen.style.opacity = '0';
            setTimeout(() => loadingScreen.style.display = 'none', 500);
        }
    };

    eventSource.onerror = function() {
        console.error("SSE Connection lost. Retrying...");
        eventSource.close();
        setTimeout(connectSSE, 3000); // Retry after 3s
    };
}

// Fetch News
async function fetchNews() {
    const container = document.getElementById('news-container');
    try {
        const response = await fetch('/api/news');
        const data = await response.json();
        
        container.innerHTML = ''; // Clear skeleton
        
        data.news.forEach(item => {
            const div = document.createElement('div');
            div.className = 'border-l-2 border-slate-700 pl-4 py-1 hover:border-yellow-500 transition-colors cursor-default group';
            div.innerHTML = `
                <div class="flex justify-between items-start mb-1">
                    <span class="text-xs text-yellow-500 font-semibold bg-yellow-900/20 px-1.5 py-0.5 rounded">${item.category}</span>
                    <span class="text-xs text-slate-500">${item.time}</span>
                </div>
                <h4 class="text-sm font-medium text-slate-300 group-hover:text-white transition-colors">${item.title}</h4>
            `;
            container.appendChild(div);
        });
    } catch (e) {
        console.error("News fetch error:", e);
    }
}

// Init
document.addEventListener('DOMContentLoaded', async () => {
    // 1. Fetch history for chart init
    try {
        const res = await fetch('/api/history');
        const data = await res.json();
        initChart(data.history);
    } catch (e) {
        initChart([]); // Fallback
    }

    // 2. Start Stream
    connectSSE();

    // 3. Start News Polling
    fetchNews();
    setInterval(fetchNews, 30000); // Every 30s
});
