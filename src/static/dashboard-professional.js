// ========== DASHBOARD PROFESSIONAL BLUE THEME - Grafana/Datadog Style ==========

// Global State
let autoRefreshInterval = null;
let isAutoRefreshEnabled = false;
let charts = {};
let currentFullscreenChart = null;

// Professional Blue Theme Colors
const THEME_COLORS = {
    // Chart Colors
    tempMax: '#f43f5e',
    tempMean: '#f97316',
    tempMin: '#06b6d4',
    humidity: '#06b6d4',
    precipitation: '#3b82f6',
    wind: '#a855f7',
    prediction: '#fbbf24',
    
    // UI Colors
    primary: '#3b82f6',
    success: '#10b981',
    warning: '#f59e0b',
    danger: '#ef4444',
    
    // Text Colors
    textPrimary: '#f8fafc',
    textSecondary: '#94a3b8',
    textMuted: '#64748b',
    
    // Background
    bgCard: '#1a2332',
    border: '#2d3748'
};

// Alias for backward compatibility
const AURORA_COLORS = THEME_COLORS;

// ========== THEME MANAGEMENT ==========
function toggleTheme() {
    const body = document.body;
    const currentTheme = body.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? null : 'light';
    
    if (newTheme) {
        body.setAttribute('data-theme', newTheme);
    } else {
        body.removeAttribute('data-theme');
    }
    
    localStorage.setItem('dashboard-theme', newTheme || 'dark');
    
    // Update button icon
    const btn = document.getElementById('themeToggle');
    if (btn) {
        btn.innerHTML = newTheme === 'light' 
            ? '<i class="fas fa-moon"></i>'
            : '<i class="fas fa-sun"></i>';
    }
    
    // Update all charts with new theme colors
    updateChartsTheme();
    
    showNotification('Theme changed successfully', 'success');
}

function updateChartsTheme() {
    const isLight = document.body.getAttribute('data-theme') === 'light';
    const textColor = isLight ? '#0f172a' : '#f1f5f9';
    const gridColor = isLight ? 'rgba(148, 163, 184, 0.3)' : 'rgba(148, 163, 184, 0.1)';
    
    Object.values(charts).forEach(chart => {
        if (chart && chart.options) {
            chart.options.plugins.legend.labels.color = textColor;
            chart.options.scales.x.ticks.color = textColor;
            chart.options.scales.y.ticks.color = textColor;
            chart.options.scales.x.grid.color = gridColor;
            chart.options.scales.y.grid.color = gridColor;
            chart.options.scales.x.title.color = textColor;
            chart.options.scales.y.title.color = textColor;
            chart.update('none');
        }
    });
}

// Load saved theme
function loadSavedTheme() {
    const savedTheme = localStorage.getItem('dashboard-theme') || 'dark';
    
    if (savedTheme === 'light') {
        document.body.setAttribute('data-theme', 'light');
    }
    
    const btn = document.getElementById('themeToggle');
    if (btn) {
        btn.innerHTML = savedTheme === 'light'
            ? '<i class="fas fa-moon"></i>'
            : '<i class="fas fa-sun"></i>';
    }
}

// ========== NOTIFICATIONS ==========
function showNotification(message, type = 'info', duration = 4000) {
    let container = document.getElementById('notificationsContainer');
    if (!container) {
        container = document.createElement('div');
        container.id = 'notificationsContainer';
        container.className = 'notifications-container';
        document.body.appendChild(container);
    }
    
    const notification = document.createElement('div');
    notification.style.cssText = `
        background: rgba(26, 35, 50, 0.98);
        backdrop-filter: blur(20px);
        border: 1px solid ${THEME_COLORS.border};
        border-left: 4px solid ${type === 'success' ? THEME_COLORS.success : type === 'error' ? THEME_COLORS.danger : type === 'warning' ? THEME_COLORS.warning : THEME_COLORS.primary};
        border-radius: 8px;
        padding: 14px 18px;
        display: flex;
        align-items: center;
        gap: 12px;
        color: ${THEME_COLORS.textPrimary};
        font-size: 0.875em;
        font-family: 'Inter', sans-serif;
        animation: slideIn 0.3s ease-out;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
        max-width: 380px;
    `;
    
    const icons = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-circle',
        warning: 'fa-exclamation-triangle',
        info: 'fa-info-circle'
    };
    
    const colors = {
        success: THEME_COLORS.success,
        error: THEME_COLORS.danger,
        warning: THEME_COLORS.warning,
        info: THEME_COLORS.primary
    };
    
    notification.innerHTML = `
        <i class="fas ${icons[type] || icons.info}" style="color: ${colors[type] || colors.info}; font-size: 1.1em;"></i>
        <span style="flex: 1;">${message}</span>
        <button onclick="this.parentElement.remove()" style="background: none; border: none; color: ${THEME_COLORS.textMuted}; cursor: pointer; padding: 4px;">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    container.appendChild(notification);
    
    setTimeout(() => {
        if (notification.parentElement) {
            notification.style.opacity = '0';
            notification.style.transform = 'translateX(100px)';
            notification.style.transition = 'all 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }
    }, duration);
}

// Add animation styles
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(100px); }
        to { opacity: 1; transform: translateX(0); }
    }
`;
document.head.appendChild(style);

// ========== REFRESH & AUTO-REFRESH ==========
async function refreshDashboard() {
    showNotification('Refreshing dashboard...', 'info', 2000);
    setTimeout(() => window.location.reload(), 500);
}

function toggleAutoRefresh() {
    isAutoRefreshEnabled = !isAutoRefreshEnabled;
    
    if (isAutoRefreshEnabled) {
        autoRefreshInterval = setInterval(() => {
            refreshDashboard();
        }, 5 * 60 * 1000); // Every 5 minutes
        showNotification('Auto-refresh enabled (5 min)', 'success');
    } else {
        clearInterval(autoRefreshInterval);
        showNotification('Auto-refresh disabled', 'info');
    }
    
    localStorage.setItem('autoRefresh', isAutoRefreshEnabled);
}

// ========== DATA COLLECTION ==========
async function triggerCollection() {
    const btn = event?.target?.closest('button');
    if (btn) {
        btn.disabled = true;
        const icon = btn.querySelector('i');
        if (icon) icon.className = 'fas fa-spinner fa-spin';
    }
    
    try {
        const response = await fetch('/api/v1/collection/trigger', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        showNotification('✅ Data collection started!', 'success', 5000);
        setTimeout(() => window.location.reload(), 3000);
        
    } catch (error) {
        console.error('Collection error:', error);
        showNotification(`❌ Error: ${error.message}`, 'error');
    } finally {
        if (btn) {
            btn.disabled = false;
            const icon = btn.querySelector('i');
            if (icon) icon.className = 'fas fa-cloud-download-alt';
        }
    }
}

// ========== EXPORT FUNCTIONALITY ==========
function exportDashboard() {
    showNotification('📊 Preparing export...', 'info');
    
    // Export chart data as JSON
    const exportData = {
        exported_at: new Date().toISOString(),
        charts: chartData,
        source: 'Climate MLOps Dashboard'
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `climate-dashboard-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
    
    showNotification('✅ Dashboard exported!', 'success');
}

// ========== CHART FUNCTIONS ==========
function getChartOptions(yAxisLabel) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            intersect: false,
            mode: 'index'
        },
        plugins: {
            legend: {
                display: true,
                position: 'top',
                align: 'end',
                labels: {
                    color: THEME_COLORS.textSecondary,
                    font: { size: 11, weight: '500', family: 'Inter' },
                    padding: 16,
                    usePointStyle: true,
                    pointStyle: 'circle',
                    boxWidth: 8
                }
            },
            tooltip: {
                backgroundColor: 'rgba(10, 14, 23, 0.95)',
                titleColor: THEME_COLORS.textPrimary,
                bodyColor: THEME_COLORS.textSecondary,
                borderColor: THEME_COLORS.primary,
                borderWidth: 1,
                padding: 12,
                cornerRadius: 8,
                titleFont: { size: 12, weight: '600', family: 'Inter' },
                bodyFont: { size: 11, family: 'Inter' },
                displayColors: true,
                boxPadding: 4
            }
        },
        scales: {
            y: {
                beginAtZero: false,
                ticks: {
                    color: THEME_COLORS.textMuted,
                    font: { size: 10, family: 'Inter' },
                    padding: 8
                },
                grid: {
                    color: 'rgba(45, 55, 72, 0.5)',
                    drawBorder: false
                },
                title: {
                    display: true,
                    text: yAxisLabel,
                    color: THEME_COLORS.textSecondary,
                    font: { size: 11, weight: '500', family: 'Inter' }
                }
            },
            x: {
                ticks: {
                    color: THEME_COLORS.textMuted,
                    font: { size: 10, family: 'Inter' },
                    maxRotation: 45,
                    minRotation: 45,
                    padding: 4
                },
                grid: {
                    color: 'rgba(45, 55, 72, 0.3)',
                    drawBorder: false
                }
            }
        },
        animation: {
            duration: 800,
            easing: 'easeOutQuart'
        }
    };
}

function createChart(chartId, chartType, labels, datasets, title, yAxisLabel) {
    const ctx = document.getElementById(chartId);
    if (!ctx) return;
    
    const isLight = document.body.getAttribute('data-theme') === 'light';
    const textColor = isLight ? '#0f172a' : '#f1f5f9';
    const gridColor = isLight ? 'rgba(148, 163, 184, 0.3)' : 'rgba(148, 163, 184, 0.1)';
    
    const config = {
        type: chartType,
        data: {
            labels: labels,
            datasets: datasets.map(ds => ({
                ...ds,
                tension: 0.4,
                fill: ds.fill !== false,
                pointRadius: ds.pointRadius || 5,
                pointHoverRadius: 8,
                pointBorderColor: 'rgba(255, 255, 255, 0.8)',
                pointBorderWidth: 2
            }))
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: textColor,
                        font: { size: 12, weight: '500', family: 'Poppins' },
                        padding: 20,
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    titleColor: '#f1f5f9',
                    bodyColor: '#94a3b8',
                    borderColor: AURORA_COLORS.primary,
                    borderWidth: 1,
                    padding: 14,
                    cornerRadius: 12,
                    titleFont: { size: 13, weight: 'bold', family: 'Poppins' },
                    bodyFont: { size: 12, family: 'Poppins' },
                    displayColors: true,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) label += ': ';
                            if (context.parsed.y !== null) {
                                label += context.parsed.y.toFixed(1);
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: chartType === 'bar',
                    ticks: {
                        color: textColor,
                        font: { size: 11, family: 'Poppins' }
                    },
                    grid: {
                        color: gridColor,
                        drawBorder: false
                    },
                    title: {
                        display: true,
                        text: yAxisLabel,
                        color: textColor,
                        font: { size: 12, weight: '600', family: 'Poppins' }
                    }
                },
                x: {
                    ticks: {
                        color: textColor,
                        font: { size: 11, family: 'Poppins' },
                        maxRotation: 45,
                        minRotation: 45
                    },
                    grid: {
                        color: gridColor,
                        drawBorder: false
                    },
                    title: {
                        display: true,
                        text: 'Date',
                        color: textColor,
                        font: { size: 12, weight: '600', family: 'Poppins' }
                    }
                }
            },
            animation: {
                duration: 1200,
                easing: 'easeOutQuart'
            }
        }
    };
    
    // Destroy existing chart if it exists
    if (charts[chartId]) {
        charts[chartId].destroy();
    }
    
    charts[chartId] = new Chart(ctx, config);
}

// ========== INITIALIZE CHARTS WITH DATA ==========
function initializeCharts() {
    if (typeof chartData === 'undefined') {
        console.warn('chartData not available');
        return;
    }
    
    let dates = [...chartData.dates];
    let tempMean = [...chartData.temperature_mean];
    let tempMax = [...chartData.temperature_max];
    let tempMin = [...chartData.temperature_min];
    let humidity = [...chartData.humidity];
    let precipitation = [...chartData.precipitation];
    let windspeed = [...chartData.windspeed];
    
    // Add today's data if available
    if (chartData.today_forecast) {
        const today = new Date().toISOString().split('T')[0];
        if (!dates.includes(today)) {
            dates.push(today);
            tempMean.push(chartData.today_forecast.temperature_mean);
            tempMax.push(chartData.today_forecast.temperature_max);
            tempMin.push(chartData.today_forecast.temperature_min);
            humidity.push(chartData.today_forecast.humidity);
            precipitation.push(chartData.today_forecast.precipitation);
            windspeed.push(chartData.today_forecast.windspeed);
        }
    }
    
    // Create highlight for today
    const createPointRadii = () => dates.map((_, idx) => idx === dates.length - 1 ? 10 : 4);
    const createPointBorderWidth = () => dates.map((_, idx) => idx === dates.length - 1 ? 3 : 2);
    
    // Temperature Chart
    createChart(
        'tempChart',
        'line',
        dates,
        [
            {
                label: 'Max Temperature',
                data: tempMax,
                borderColor: AURORA_COLORS.tempMax,
                backgroundColor: `${AURORA_COLORS.tempMax}20`,
                borderWidth: 3,
                pointRadius: createPointRadii(),
                pointBackgroundColor: AURORA_COLORS.tempMax,
                pointBorderWidth: createPointBorderWidth()
            },
            {
                label: 'Mean Temperature',
                data: tempMean,
                borderColor: AURORA_COLORS.tempMean,
                backgroundColor: `${AURORA_COLORS.tempMean}20`,
                borderWidth: 3,
                pointRadius: createPointRadii(),
                pointBackgroundColor: AURORA_COLORS.tempMean,
                pointBorderWidth: createPointBorderWidth()
            },
            {
                label: 'Min Temperature',
                data: tempMin,
                borderColor: AURORA_COLORS.tempMin,
                backgroundColor: `${AURORA_COLORS.tempMin}20`,
                borderWidth: 3,
                pointRadius: createPointRadii(),
                pointBackgroundColor: AURORA_COLORS.tempMin,
                pointBorderWidth: createPointBorderWidth()
            },
            {
                label: '🤖 AI Prediction',
                data: dates.map((_, idx) => 
                    idx === dates.length - 1 ? chartData.prediction.temperature_mean : null
                ),
                borderColor: AURORA_COLORS.prediction,
                backgroundColor: AURORA_COLORS.prediction,
                borderWidth: 4,
                pointRadius: 12,
                pointStyle: 'star',
                showLine: false
            }
        ],
        'Temperature Trends',
        'Temperature (°C)'
    );
    
    // Humidity Chart
    createChart(
        'humidityChart',
        'line',
        dates,
        [
            {
                label: 'Relative Humidity',
                data: humidity,
                borderColor: AURORA_COLORS.humidity,
                backgroundColor: `${AURORA_COLORS.humidity}20`,
                borderWidth: 3,
                pointRadius: createPointRadii(),
                pointBackgroundColor: AURORA_COLORS.humidity,
                pointBorderWidth: createPointBorderWidth()
            }
        ],
        'Relative Humidity',
        'Humidity (%)'
    );
    
    // Precipitation Chart
    createChart(
        'precipChart',
        'bar',
        dates,
        [
            {
                label: 'Precipitation',
                data: precipitation,
                borderColor: AURORA_COLORS.precipitation,
                backgroundColor: dates.map((_, idx) => 
                    idx === dates.length - 1 ? `${AURORA_COLORS.prediction}cc` : `${AURORA_COLORS.precipitation}aa`
                ),
                borderWidth: 2,
                borderRadius: 6
            }
        ],
        'Precipitation',
        'Precipitation (mm)'
    );
    
    // Wind Chart
    createChart(
        'windChart',
        'line',
        dates,
        [
            {
                label: 'Max Wind Speed',
                data: windspeed,
                borderColor: AURORA_COLORS.wind,
                backgroundColor: `${AURORA_COLORS.wind}20`,
                borderWidth: 3,
                pointRadius: createPointRadii(),
                pointBackgroundColor: AURORA_COLORS.wind,
                pointBorderWidth: createPointBorderWidth()
            }
        ],
        'Wind Speed',
        'Speed (km/h)'
    );
}

// ========== CHART DOWNLOAD ==========
function downloadChart(chartId) {
    const chart = charts[chartId];
    if (!chart) return;
    
    const url = chart.toBase64Image();
    const link = document.createElement('a');
    link.href = url;
    link.download = `${chartId}-${new Date().toISOString().split('T')[0]}.png`;
    link.click();
    
    showNotification('Chart downloaded successfully', 'success');
}

// ========== FILTER UPDATES ==========
function updateDashboard() {
    const period = document.getElementById('periodFilter')?.value || '7';
    const metric = document.getElementById('metricFilter')?.value || 'all';
    
    try {
        const filteredData = filterDataByPeriod(parseInt(period));
        const finalData = filterDataByMetric(filteredData, metric);
        updateChartsWithData(finalData);
        showNotification(`Filters applied: ${period} days`, 'success');
    } catch (error) {
        console.error('Filter error:', error);
        showNotification(`Filter error: ${error.message}`, 'error');
    }
}

function filterDataByPeriod(days) {
    if (typeof chartData === 'undefined') return null;
    
    const totalDays = chartData.dates.length;
    const startIndex = Math.max(0, totalDays - days);
    
    return {
        dates: chartData.dates.slice(startIndex),
        temperature_mean: chartData.temperature_mean.slice(startIndex),
        temperature_max: chartData.temperature_max.slice(startIndex),
        temperature_min: chartData.temperature_min.slice(startIndex),
        humidity: chartData.humidity.slice(startIndex),
        precipitation: chartData.precipitation.slice(startIndex),
        windspeed: chartData.windspeed.slice(startIndex),
        prediction: chartData.prediction
    };
}

function filterDataByMetric(data, metric) {
    if (!data) return null;
    return { ...data, metric };
}

function updateChartsWithData(filteredData) {
    if (!filteredData) return;
    
    // Reinitialize charts with filtered data
    const tempChartData = { ...chartData, ...filteredData };
    const originalChartData = chartData;
    window.chartData = tempChartData;
    initializeCharts();
    window.chartData = originalChartData;
}

// ========== INITIALIZE ON DOM LOAD ==========
document.addEventListener('DOMContentLoaded', function() {
    loadSavedTheme();
    
    setTimeout(() => {
        initializeCharts();
    }, 300);
    
    const periodFilter = document.getElementById('periodFilter');
    const metricFilter = document.getElementById('metricFilter');
    
    if (periodFilter) periodFilter.addEventListener('change', updateDashboard);
    if (metricFilter) metricFilter.addEventListener('change', updateDashboard);
    
    const autoRefreshSaved = localStorage.getItem('autoRefresh') === 'true';
    if (autoRefreshSaved) {
        isAutoRefreshEnabled = true;
        toggleAutoRefresh();
    }
    
    setTimeout(() => {
        showNotification('Dashboard loaded successfully! 🌤️', 'success', 3000);
    }, 800);
    
    console.log('🌤️ Climate MLOps Dashboard - Aurora Theme loaded');
});

// ========== UTILITIES ==========
function formatDate(date) {
    return new Date(date).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

function formatNumber(num, decimals = 1) {
    return Number(num).toFixed(decimals);
}