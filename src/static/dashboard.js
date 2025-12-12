// dashboard.js - Interactive features for Climate Dashboard

// Function to refresh the dashboard
function refreshDashboard() {
    const loading = document.getElementById('loading');
    const refreshBtn = document.querySelector('.refresh-btn');
    
    // Show loading spinner
    loading.style.display = 'block';
    refreshBtn.disabled = true;
    refreshBtn.textContent = 'Actualisation...';
    
    // Simulate loading time (in real app, this would be AJAX)
    setTimeout(() => {
        // Reload the page to get fresh data
        window.location.reload();
    }, 1000);
}

// Function to toggle dark/light theme
function toggleTheme() {
    const body = document.body;
    const themeToggle = document.querySelector('.theme-toggle i');
    
    if (body.getAttribute('data-theme') === 'dark') {
        body.removeAttribute('data-theme');
        themeToggle.className = 'fas fa-moon';
        localStorage.setItem('theme', 'light');
    } else {
        body.setAttribute('data-theme', 'dark');
        themeToggle.className = 'fas fa-sun';
        localStorage.setItem('theme', 'dark');
    }
    
    // Update Chart.js theme
    updateChartTheme();
}

// Function to update Chart.js theme
function updateChartTheme() {
    const isDark = document.body.getAttribute('data-theme') === 'dark';
    const textColor = isDark ? '#e0e0e0' : '#333';
    const gridColor = isDark ? '#555' : '#ddd';
    
    // Update all existing charts
    Chart.instances.forEach(chart => {
        chart.options.plugins.legend.labels.color = textColor;
        chart.options.scales.x.ticks.color = textColor;
        chart.options.scales.y.ticks.color = textColor;
        chart.options.scales.x.grid.color = gridColor;
        chart.options.scales.y.grid.color = gridColor;
        chart.update();
    });
}

// Load theme from localStorage
function loadTheme() {
    const savedTheme = localStorage.getItem('theme');
    const themeToggle = document.querySelector('.theme-toggle i');
    
    if (savedTheme === 'dark') {
        document.body.setAttribute('data-theme', 'dark');
        themeToggle.className = 'fas fa-sun';
    } else {
        themeToggle.className = 'fas fa-moon';
    }
}

// Add tooltips to weather items
function initTooltips() {
    const weatherItems = document.querySelectorAll('.weather-item');
    
    weatherItems.forEach(item => {
        const tooltip = item.getAttribute('data-tooltip');
        if (tooltip) {
            item.setAttribute('title', tooltip);
        }
    });
}

// Animate cards on scroll
function animateOnScroll() {
    const cards = document.querySelectorAll('.card');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, { threshold: 0.1 });
    
    cards.forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(card);
    });
}

// Add click effects to weather items
function addClickEffects() {
    const weatherItems = document.querySelectorAll('.weather-item');
    
    weatherItems.forEach(item => {
        item.addEventListener('click', () => {
            // Add a pulse effect
            item.style.animation = 'pulse 0.3s ease';
            setTimeout(() => {
                item.style.animation = '';
            }, 300);
        });
    });
}

// Create temperature chart
function createTempChart() {
    const ctx = document.getElementById('tempChart').getContext('2d');
    
    const todayIndex = chartData.dates.length - 1; // Last day is today
    const isDark = document.body.getAttribute('data-theme') === 'dark';
    const textColor = isDark ? '#e0e0e0' : '#333';
    const gridColor = isDark ? '#555' : '#ddd';
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.dates,
            datasets: [
                {
                    label: 'Temp Moyenne Historique',
                    data: chartData.temperature_mean,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4,
                    fill: false
                },
                {
                    label: 'Temp Max Historique',
                    data: chartData.temperature_max,
                    borderColor: '#ff6b6b',
                    backgroundColor: 'rgba(255, 107, 107, 0.1)',
                    tension: 0.4,
                    fill: false
                },
                {
                    label: 'Temp Min Historique',
                    data: chartData.temperature_min,
                    borderColor: '#4ecdc4',
                    backgroundColor: 'rgba(78, 205, 196, 0.1)',
                    tension: 0.4,
                    fill: false
                },
                {
                    label: 'Prévision Aujourd\'hui',
                    data: chartData.dates.map((date, index) => 
                        index === todayIndex ? chartData.today_forecast.temperature_mean : null
                    ),
                    borderColor: '#ffa726',
                    backgroundColor: '#ffa726',
                    pointRadius: 8,
                    pointHoverRadius: 10,
                    type: 'scatter',
                    showLine: false
                },
                {
                    label: 'Prédiction Modèle',
                    data: chartData.dates.map((date, index) => 
                        index === todayIndex ? chartData.prediction.temperature_mean : null
                    ),
                    borderColor: '#ab47bc',
                    backgroundColor: '#ab47bc',
                    pointRadius: 8,
                    pointHoverRadius: 10,
                    type: 'scatter',
                    showLine: false
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    labels: {
                        color: textColor
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: 'Température (°C)',
                        color: textColor
                    },
                    ticks: {
                        color: textColor
                    },
                    grid: {
                        color: gridColor
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Date',
                        color: textColor
                    },
                    ticks: {
                        color: textColor
                    },
                    grid: {
                        color: gridColor
                    }
                }
            }
        }
    });
}

// Create humidity chart
function createHumidityChart() {
    const ctx = document.getElementById('humidityChart').getContext('2d');
    
    const todayIndex = chartData.dates.length - 1;
    const isDark = document.body.getAttribute('data-theme') === 'dark';
    const textColor = isDark ? '#e0e0e0' : '#333';
    const gridColor = isDark ? '#555' : '#ddd';
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.dates,
            datasets: [
                {
                    label: 'Humidité Relative',
                    data: chartData.humidity,
                    borderColor: '#26a69a',
                    backgroundColor: 'rgba(38, 166, 154, 0.1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Prévision Aujourd\'hui',
                    data: chartData.dates.map((date, index) => 
                        index === todayIndex ? chartData.today_forecast.humidity : null
                    ),
                    borderColor: '#ffa726',
                    backgroundColor: '#ffa726',
                    pointRadius: 8,
                    pointHoverRadius: 10,
                    type: 'scatter',
                    showLine: false
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    labels: {
                        color: textColor
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Humidité (%)',
                        color: textColor
                    },
                    ticks: {
                        color: textColor
                    },
                    grid: {
                        color: gridColor
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Date',
                        color: textColor
                    },
                    ticks: {
                        color: textColor
                    },
                    grid: {
                        color: gridColor
                    }
                }
            }
        }
    });
}

// Create precipitation chart
function createPrecipChart() {
    const ctx = document.getElementById('precipChart').getContext('2d');
    
    const todayIndex = chartData.dates.length - 1;
    const isDark = document.body.getAttribute('data-theme') === 'dark';
    const textColor = isDark ? '#e0e0e0' : '#333';
    const gridColor = isDark ? '#555' : '#ddd';
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: chartData.dates,
            datasets: [
                {
                    label: 'Précipitations',
                    data: chartData.precipitation,
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Prévision Aujourd\'hui',
                    data: chartData.dates.map((date, index) => 
                        index === todayIndex ? chartData.today_forecast.precipitation : null
                    ),
                    backgroundColor: '#ffa726',
                    borderColor: '#ffa726',
                    type: 'bar'
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    labels: {
                        color: textColor
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Précipitations (mm)',
                        color: textColor
                    },
                    ticks: {
                        color: textColor
                    },
                    grid: {
                        color: gridColor
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Date',
                        color: textColor
                    },
                    ticks: {
                        color: textColor
                    },
                    grid: {
                        color: gridColor
                    }
                }
            }
        }
    });
}

// Create wind speed chart
function createWindChart() {
    const ctx = document.getElementById('windChart').getContext('2d');
    
    const todayIndex = chartData.dates.length - 1;
    const isDark = document.body.getAttribute('data-theme') === 'dark';
    const textColor = isDark ? '#e0e0e0' : '#333';
    const gridColor = isDark ? '#555' : '#ddd';
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.dates,
            datasets: [
                {
                    label: 'Vitesse du Vent Max',
                    data: chartData.windspeed,
                    borderColor: '#8e24aa',
                    backgroundColor: 'rgba(142, 36, 170, 0.1)',
                    tension: 0.4,
                    fill: false
                },
                {
                    label: 'Prévision Aujourd\'hui',
                    data: chartData.dates.map((date, index) => 
                        index === todayIndex ? chartData.today_forecast.windspeed : null
                    ),
                    borderColor: '#ffa726',
                    backgroundColor: '#ffa726',
                    pointRadius: 8,
                    pointHoverRadius: 10,
                    type: 'scatter',
                    showLine: false
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    labels: {
                        color: textColor
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Vitesse du Vent (km/h)',
                        color: textColor
                    },
                    ticks: {
                        color: textColor
                    },
                    grid: {
                        color: gridColor
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Date',
                        color: textColor
                    },
                    ticks: {
                        color: textColor
                    },
                    grid: {
                        color: gridColor
                    }
                }
            }
        }
    });
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    loadTheme();
    initTooltips();
    animateOnScroll();
    addClickEffects();
    
    // Create charts if data is available
    if (typeof chartData !== 'undefined') {
        createTempChart();
        createHumidityChart();
        createPrecipChart();
        createWindChart();
    }
    
    // Add some CSS animations
    const style = document.createElement('style');
    style.textContent = `
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
        
        .weather-item {
            user-select: none;
        }
        
        .chart-container canvas {
            max-width: 100%;
            height: auto !important;
        }
    `;
    document.head.appendChild(style);
});

// Add keyboard shortcut for refresh (Ctrl+R is default, but we can add Shift+R)
document.addEventListener('keydown', (e) => {
    if (e.shiftKey && e.key === 'R') {
        e.preventDefault();
        refreshDashboard();
    }
});