// Load analysis results from localStorage on page load
window.addEventListener('DOMContentLoaded', function () {
    const analysisData = localStorage.getItem('currentAnalysis');

    if (analysisData) {
        try {
            const data = JSON.parse(analysisData);
            console.log('Loaded analysis data:', data);

            // Update uploaded image
            const imageElement = document.querySelector('.lg\\:col-span-3 > div');
            if (data.image && imageElement) {
                imageElement.style.backgroundImage = `url(${data.image})`;
            }

            // Update disease name
            const diseaseNameElement = document.querySelector('h1.text-2xl');
            if (diseaseNameElement && data.diseaseName) {
                const words = data.diseaseName.split(' ');
                if (words.length > 1) {
                    diseaseNameElement.innerHTML = `${words[0]} <span class="gradient-text">${words.slice(1).join(' ')}</span>`;
                } else {
                    diseaseNameElement.innerHTML = `<span class="gradient-text">${data.diseaseName}</span>`;
                }
            }

            // Update crop type
            const cropTypeElement = diseaseNameElement ? diseaseNameElement.nextElementSibling : null;
            if (cropTypeElement && data.cropName) {
                cropTypeElement.textContent = data.cropName;
            }

            // Update severity level text and percentage
            const severityTextElement = document.querySelector('.flex.justify-between.items-end span:last-child');
            if (severityTextElement && data.severity) {
                // Extract numeric severity if available
                let severityPercent = '50';
                if (data.confidencePercent) {
                    severityPercent = data.confidencePercent.replace('%', '');
                }
                severityTextElement.textContent = `${data.severity} (${severityPercent}%)`;
            }

            // Update severity bar width
            const severityBar = document.querySelector('.severity-bar');
            if (severityBar && data.confidencePercent) {
                const percent = data.confidencePercent.replace('%', '');
                severityBar.style.width = `${percent}%`;
            }

            // Update confidence percentage
            const confidenceElement = document.querySelector('.stat-card:nth-child(1) .stat-value');
            if (confidenceElement && data.confidencePercent) {
                confidenceElement.textContent = data.confidencePercent;
            }

            // Update severity stat
            const severityStatElement = document.querySelector('.stat-card:nth-child(2) .stat-value');
            if (severityStatElement && data.confidencePercent) {
                severityStatElement.textContent = data.confidencePercent;
            }

            // Update confidence badge on image
            const confidenceBadge = document.querySelector('.pulse-box > div');
            if (confidenceBadge && data.confidencePercent) {
                confidenceBadge.textContent = data.confidencePercent;
            }

            // Update AI recommendations if available
            if (data.recommendations && data.recommendations.length > 0) {
                const actionsContainer = document.querySelector('.lg\\:col-span-5 .grid');
                if (actionsContainer) {
                    // Clear existing static recommendations (keep first 4 as fallback)
                    // Update with AI recommendations
                    let html = '';

                    data.recommendations.forEach((rec, index) => {
                        // Color mapping
                        const colors = [
                            'from-blue-500/20 to-blue-600/20 text-blue-400',
                            'from-primary/20 to-accent/20 text-primary',
                            'from-rose-500/20 to-rose-600/20 text-rose-400',
                            'from-yellow-500/20 to-yellow-600/20 text-yellow-400',
                            'from-purple-500/20 to-purple-600/20 text-purple-400'
                        ];

                        const colorClass = colors[index % colors.length];
                        const icon = rec.icon || 'agriculture';

                        html += `
                            <div class="action-card group flex items-center gap-3 rounded-xl glass p-3 cursor-pointer">
                                <div class="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br ${colorClass}">
                                    <span class="material-symbols-outlined text-[22px]">${icon}</span>
                                </div>
                                <div class="flex-1 min-w-0">
                                    <h4 class="text-sm font-bold text-white">${rec.title}</h4>
                                    <p class="text-xs text-gray-400 truncate">${rec.description}</p>
                                </div>
                                <span class="material-symbols-outlined text-gray-600 group-hover:text-primary transition-colors text-lg">chevron_right</span>
                            </div>
                        `;
                    });

                    actionsContainer.innerHTML = html;
                    console.log('AI recommendations updated!');
                }
            }

            console.log('UI updated with real data!');
        } catch (error) {
            console.error('Error loading analysis data:', error);
        }
    } else {
        console.warn('No analysis data found in localStorage');
    }
});
