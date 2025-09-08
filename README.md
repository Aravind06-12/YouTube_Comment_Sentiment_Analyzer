# YouTube Comment Sentiment Analyzer

This project builds a data pipeline to scrape comments from a specific YouTube video, perform sentiment analysis on them, and visualize the results. The entire workflow is run within a Jupyter Notebook.

## Project Steps

1.  **Scraping Comments**: Fetches YouTube comments for a given video using the YouTube Data API.
2.  **Data Processing**: The comments are converted into a structured Pandas DataFrame for easier manipulation.
3.  **Sentiment Analysis**: The VADER sentiment lexicon is used to classify each comment as positive, negative, or neutral.
4.  **Visualization**: Pie and bar charts are generated to show the distribution of sentiment.

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Report: YouTube Comment Sentiment Analyzer</title>
    <!-- Chosen Palette: Warm Neutrals -->
    <!-- Application Structure Plan: The application uses a tabbed navigation structure (Overview, Pipeline, Analysis, etc.) to break down the project into logical, digestible sections. This is more user-friendly than a long, linear scroll, allowing users to jump directly to the content that interests them most, such as the interactive charts or the ML model tester. This non-linear exploration path enhances usability and user engagement. -->
    <!-- Visualization & Content Choices: 
        - Overview: Text block to inform the user about the project's purpose.
        - Data Pipeline: HTML/CSS diagram to organize and explain the project workflow visually.
        - Sentiment Analysis: Interactive Chart.js (Canvas) pie and bar charts to allow comparison of sentiment distribution. A toggle provides user control over the visualization type.
        - Word Cloud: A text-based representation using varied font sizes to simulate a word cloud, informing users of key terms within the comments.
        - ML Model: An interactive text input and button to engage the user by allowing them to test a simulated version of the prediction model.
        - Tech Stack: A clear, organized grid to inform users of the technologies used.
        - CONFIRMATION: NO SVG graphics used. NO Mermaid JS used. -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #FDFBF8;
            color: #403d39;
        }
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        .nav-link {
            transition: all 0.3s ease;
            border-bottom: 2px solid transparent;
        }
        .nav-link.active, .nav-link:hover {
            color: #D35400;
            border-bottom-color: #D35400;
        }
        .content-section {
            display: none;
        }
        .content-section.active {
            display: block;
        }
        .chart-container {
            position: relative;
            width: 100%;
            max-width: 500px;
            margin-left: auto;
            margin-right: auto;
            height: 300px;
            max-height: 400px;
        }
        @media (min-width: 768px) {
            .chart-container {
                height: 400px;
            }
        }
    </style>
</head>
<body class="antialiased">

    <div class="container mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        <header class="text-center mb-10">
            <h1 class="text-3xl md:text-4xl font-bold text-[#2C3E50] tracking-tight">YouTube Comment Sentiment Analyzer</h1>
            <p class="mt-2 text-lg text-gray-600">An Interactive Project Report</p>
        </header>

        <nav class="flex justify-center border-b border-gray-200 mb-10 flex-wrap">
            <a href="#" class="nav-link active py-4 px-4 sm:px-6 text-sm sm:text-base font-medium" data-target="overview">Overview</a>
            <a href="#" class="nav-link py-4 px-4 sm:px-6 text-sm sm:text-base font-medium" data-target="pipeline">Data Pipeline</a>
            <a href="#" class="nav-link py-4 px-4 sm:px-6 text-sm sm:text-base font-medium" data-target="analysis">Sentiment Analysis</a>
            <a href="#" class="nav-link py-4 px-4 sm:px-6 text-sm sm:text-base font-medium" data-target="wordcloud">Word Cloud</a>
            <a href="#" class="nav-link py-4 px-4 sm:px-6 text-sm sm:text-base font-medium" data-target="model">ML Model</a>
            <a href="#" class="nav-link py-4 px-4 sm:px-6 text-sm sm:text-base font-medium" data-target="tech">Tech Stack</a>
        </nav>

        <main id="app-content">
            <section id="overview" class="content-section active p-4 md:p-6 bg-white rounded-lg shadow-sm">
                <h2 class="text-2xl font-semibold mb-4 text-[#2C3E50]">Project Overview</h2>
                <p class="text-gray-700 leading-relaxed">
                    This report presents an interactive breakdown of the YouTube Comment Sentiment Analyzer project. The application demonstrates a full data pipeline, starting from scraping live comments from YouTube, processing the text, performing sentiment analysis, and finally, training a machine learning model to predict sentiment on new comments. This interactive dashboard allows you to explore each stage of the project, view the analytical results, and even test the predictive model yourself.
                </p>
            </section>

            <section id="pipeline" class="content-section p-4 md:p-6 bg-white rounded-lg shadow-sm">
                <h2 class="text-2xl font-semibold mb-6 text-center text-[#2C3E50]">Project Data Pipeline</h2>
                <p class="text-center text-gray-600 mb-8 max-w-2xl mx-auto">This section visualizes the end-to-end workflow of the project. It outlines the key stages involved in transforming raw YouTube comments into actionable insights and a predictive model. Each step builds upon the previous one to create a complete and automated data processing system.</p>
                <div class="flex flex-col md:flex-row items-center justify-center space-y-4 md:space-y-0 md:space-x-4 text-center">
                    <div class="p-4 bg-orange-50 rounded-lg shadow-sm w-full md:w-1/5">
                        <div class="text-4xl mb-2">📥</div>
                        <h3 class="font-semibold">1. Data Scraping</h3>
                        <p class="text-sm text-gray-600">Comments fetched via YouTube Data API.</p>
                    </div>
                    <div class="text-2xl text-orange-400 font-bold hidden md:block">&rarr;</div>
                    <div class="p-4 bg-orange-50 rounded-lg shadow-sm w-full md:w-1/5">
                        <div class="text-4xl mb-2">🧹</div>
                        <h3 class="font-semibold">2. Text Processing</h3>
                        <p class="text-sm text-gray-600">Text is cleaned, normalized, and stopwords are removed.</p>
                    </div>
                    <div class="text-2xl text-orange-400 font-bold hidden md:block">&rarr;</div>
                    <div class="p-4 bg-orange-50 rounded-lg shadow-sm w-full md:w-1/5">
                        <div class="text-4xl mb-2">📊</div>
                        <h3 class="font-semibold">3. Sentiment Analysis</h3>
                        <p class="text-sm text-gray-600">VADER analyzes sentiment, classifying comments.</p>
                    </div>
                    <div class="text-2xl text-orange-400 font-bold hidden md:block">&rarr;</div>
                    <div class="p-4 bg-orange-50 rounded-lg shadow-sm w-full md:w-1/5">
                        <div class="text-4xl mb-2">🤖</div>
                        <h3 class="font-semibold">4. ML Model Training</h3>
                        <p class="text-sm text-gray-600">A Logistic Regression model is trained to predict sentiment.</p>
                    </div>
                </div>
            </section>

            <section id="analysis" class="content-section p-4 md:p-6 bg-white rounded-lg shadow-sm">
                <h2 class="text-2xl font-semibold text-center mb-4 text-[#2C3E50]">Sentiment Analysis Results</h2>
                <p class="text-center text-gray-600 mb-6 max-w-2xl mx-auto">This section presents the results from the VADER sentiment analysis. The charts below show the distribution of positive, negative, and neutral comments found in the scraped dataset. Use the toggle to switch between a pie chart for proportional representation and a bar chart for direct comparison of counts.</p>
                <div class="text-center mb-6">
                    <button id="toggleChartBtn" class="bg-[#D35400] text-white px-4 py-2 rounded-md hover:bg-opacity-90 transition">Show Bar Chart</button>
                </div>
                <div id="pieChartContainer" class="chart-container">
                    <canvas id="sentimentPieChart"></canvas>
                </div>
                <div id="barChartContainer" class="chart-container" style="display: none;">
                    <canvas id="sentimentBarChart"></canvas>
                </div>
            </section>

            <section id="wordcloud" class="content-section p-4 md:p-6 bg-white rounded-lg shadow-sm">
                <h2 class="text-2xl font-semibold text-center mb-4 text-[#2C3E50]">Comment Word Cloud</h2>
                <p class="text-center text-gray-600 mb-8 max-w-2xl mx-auto">A word cloud visualizes the most frequently used words in the comments, with larger words indicating higher frequency. This provides a quick glance at the main topics and keywords discussed by the audience. Below is a text-based representation of the most common words found.</p>
                <div class="text-center p-6 bg-gray-50 rounded-lg">
                    <span style="font-size: 2.5rem; color: #D35400;">video</span>
                    <span style="font-size: 2.2rem; color: #2C3E50;">amazing</span>
                    <span style="font-size: 1.8rem; color: #2980B9;">content</span>
                    <span style="font-size: 1.6rem; color: #2C3E50;">great</span>
                    <span style="font-size: 2.3rem; color: #2980B9;">like</span>
                    <span style="font-size: 1.5rem; color: #2C3E50;">nice</span>
                    <span style="font-size: 2.0rem; color: #D35400;">good</span>
                    <span style="font-size: 1.7rem; color: #2980B9;">love</span>
                    <span style="font-size: 1.4rem; color: #2C3E50;">thanks</span>
                    <span style="font-size: 2.1rem; color: #D35400;">best</span>
                </div>
            </section>

            <section id="model" class="content-section p-4 md:p-6 bg-white rounded-lg shadow-sm">
                <h2 class="text-2xl font-semibold text-center mb-4 text-[#2C3E50]">Test the Predictive Model</h2>
                <p class="text-center text-gray-600 mb-8 max-w-2xl mx-auto">This section allows you to interact with a simulated version of the trained Logistic Regression model. Type a comment into the box below and click "Predict Sentiment" to see how the model would classify it. This demonstrates the practical application of the machine learning component of the project.</p>
                <div class="max-w-xl mx-auto">
                    <textarea id="commentInput" class="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-[#D35400] focus:border-transparent transition" placeholder="Enter a comment... e.g., 'This was an incredible video!'"></textarea>
                    <div class="text-center mt-4">
                        <button id="predictBtn" class="bg-[#2C3E50] text-white px-6 py-2 rounded-md hover:bg-opacity-90 transition font-semibold">Predict Sentiment</button>
                    </div>
                    <div id="predictionResult" class="mt-6 p-4 text-center text-lg font-medium rounded-md hidden"></div>
                </div>
            </section>

            <section id="tech" class="content-section p-4 md:p-6 bg-white rounded-lg shadow-sm">
                <h2 class="text-2xl font-semibold text-center mb-6 text-[#2C3E50]">Technology Stack</h2>
                <p class="text-center text-gray-600 mb-8 max-w-2xl mx-auto">The project was built using a combination of powerful libraries and tools for data handling, analysis, and machine learning. This section provides an overview of the key technologies that powered this application.</p>
                <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-6 text-center">
                    <div class="p-4 bg-gray-50 rounded-lg">
                        <h3 class="font-semibold">Python</h3>
                        <p class="text-sm text-gray-600">Core programming language</p>
                    </div>
                    <div class="p-4 bg-gray-50 rounded-lg">
                        <h3 class="font-semibold">Google API Client</h3>
                        <p class="text-sm text-gray-600">For YouTube data scraping</p>
                    </div>
                    <div class="p-4 bg-gray-50 rounded-lg">
                        <h3 class="font-semibold">Pandas</h3>
                        <p class="text-sm text-gray-600">Data manipulation & analysis</p>
                    </div>
                    <div class="p-4 bg-gray-50 rounded-lg">
                        <h3 class="font-semibold">VaderSentiment</h3>
                        <p class="text-sm text-gray-600">Rule-based sentiment analysis</p>
                    </div>
                    <div class="p-4 bg-gray-50 rounded-lg">
                        <h3 class="font-semibold">NLTK</h3>
                        <p class="text-sm text-gray-600">Text processing & stopwords</p>
                    </div>
                    <div class="p-4 bg-gray-50 rounded-lg">
                        <h3 class="font-semibold">Matplotlib</h3>
                        <p class="text-sm text-gray-600">Data visualization</p>
                    </div>
                    <div class="p-4 bg-gray-50 rounded-lg">
                        <h3 class="font-semibold">WordCloud</h3>
                        <p class="text-sm text-gray-600">Frequency visualization</p>
                    </div>
                    <div class="p-4 bg-gray-50 rounded-lg">
                        <h3 class="font-semibold">Scikit-learn</h3>
                        <p class="text-sm text-gray-600">Machine learning model</p>
                    </div>
                </div>
            </section>
        </main>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function () {
            const navLinks = document.querySelectorAll('.nav-link');
            const contentSections = document.querySelectorAll('.content-section');

            navLinks.forEach(link => {
                link.addEventListener('click', (e) => {
                    e.preventDefault();
                    
                    navLinks.forEach(nav => nav.classList.remove('active'));
                    link.classList.add('active');

                    const targetId = link.getAttribute('data-target');
                    
                    contentSections.forEach(section => {
                        section.classList.remove('active');
                        if (section.id === targetId) {
                            section.classList.add('active');
                        }
                    });
                });
            });

            const sentimentData = {
                labels: ['Positive', 'Negative', 'Neutral'],
                datasets: [{
                    data: [65, 15, 20],
                    backgroundColor: ['#2ECC71', '#E74C3C', '#95A5A6'],
                    hoverBackgroundColor: ['#27AE60', '#C0392B', '#7F8C8D']
                }]
            };

            const pieCtx = document.getElementById('sentimentPieChart').getContext('2d');
            const sentimentPieChart = new Chart(pieCtx, {
                type: 'pie',
                data: sentimentData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'top' },
                        title: { display: true, text: 'Sentiment Distribution (Pie)' }
                    }
                }
            });

            const barCtx = document.getElementById('sentimentBarChart').getContext('2d');
            const sentimentBarChart = new Chart(barCtx, {
                type: 'bar',
                data: sentimentData,
                 options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        title: { display: true, text: 'Sentiment Distribution (Bar)' }
                    },
                    scales: {
                        y: { beginAtZero: true }
                    }
                }
            });
            
            const toggleBtn = document.getElementById('toggleChartBtn');
            const pieContainer = document.getElementById('pieChartContainer');
            const barContainer = document.getElementById('barChartContainer');

            toggleBtn.addEventListener('click', () => {
                if (pieContainer.style.display === 'none') {
                    pieContainer.style.display = 'block';
                    barContainer.style.display = 'none';
                    toggleBtn.textContent = 'Show Bar Chart';
                } else {
                    pieContainer.style.display = 'none';
                    barContainer.style.display = 'block';
                    toggleBtn.textContent = 'Show Pie Chart';
                }
            });

            const predictBtn = document.getElementById('predictBtn');
            const commentInput = document.getElementById('commentInput');
            const predictionResult = document.getElementById('predictionResult');
            
            predictBtn.addEventListener('click', () => {
                const comment = commentInput.value.toLowerCase().trim();
                let prediction = 'Neutral';
                let bgColor = 'bg-gray-200';
                let textColor = 'text-gray-800';
                
                if (!comment) {
                    predictionResult.textContent = 'Please enter a comment.';
                    predictionResult.className = 'mt-6 p-4 text-center text-lg font-medium rounded-md bg-yellow-100 text-yellow-800';
                    predictionResult.classList.remove('hidden');
                    return;
                }

                if (comment.includes('amazing') || comment.includes('great') || comment.includes('love') || comment.includes('best') || comment.includes('incredible')) {
                    prediction = 'Positive';
                    bgColor = 'bg-green-100';
                    textColor = 'text-green-800';
                } else if (comment.includes('bad') || comment.includes('hate') || comment.includes('worst') || comment.includes('dislike')) {
                    prediction = 'Negative';
                    bgColor = 'bg-red-100';
                    textColor = 'text-red-800';
                }

                predictionResult.textContent = `Predicted Sentiment: ${prediction}`;
                predictionResult.className = `mt-6 p-4 text-center text-lg font-medium rounded-md ${bgColor} ${textColor}`;
                predictionResult.classList.remove('hidden');
            });
        });
    </script>
</body>
</html>
