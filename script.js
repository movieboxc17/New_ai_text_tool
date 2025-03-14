document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const inputText = document.getElementById('input-text');
    const charCount = document.getElementById('char-count');
    const summarizeBtn = document.getElementById('summarize-btn');
    const outputText = document.getElementById('output-text');
    const copyBtn = document.getElementById('copy-btn');
    const loading = document.getElementById('loading');
    const modelStatus = document.querySelector('.model-status');
    const summaryLengthSelect = document.getElementById('summary-length');

    // Global variables
    let model = null;
    let isModelLoading = false;

    // Load the Universal Sentence Encoder model
    async function loadModel() {
        if (model !== null) return model;
        if (isModelLoading) return null;

        isModelLoading = true;
        modelStatus.textContent = 'Loading AI model...';
        
        try {
            model = await use.load();
            modelStatus.textContent = 'AI model loaded successfully!';
            return model;
        } catch (error) {
            console.error('Error loading model:', error);
            modelStatus.textContent = 'Error loading AI model. Please refresh the page.';
            throw error;
        } finally {
            isModelLoading = false;
        }
    }

    // Start loading the model when the page loads
    loadModel().catch(console.error);

    // Update character count and enable/disable summarize button
    inputText.addEventListener('input', () => {
        const text = inputText.value;
        charCount.textContent = text.length;
        
        // Enable button if text is at least 100 characters
        if (text.length >= 100) {
            summarizeBtn.disabled = false;
        } else {
            summarizeBtn.disabled = true;
        }
    });

    // Summarize text when button is clicked
    summarizeBtn.addEventListener('click', async () => {
        const text = inputText.value.trim();
        
        if (text.length < 100) {
            alert('Please enter at least 100 characters to summarize.');
            return;
        }
        
        // Show loading spinner
        loading.classList.remove('hidden');
        outputText.classList.add('hidden');
        copyBtn.classList.add('hidden');
        
        try {
            // Make sure model is loaded
            if (model === null) {
                model = await loadModel();
            }
            
            const summaryRatio = parseFloat(summaryLengthSelect.value);
            const summary = await generateSummary(text, summaryRatio);
            outputText.textContent = summary;
            copyBtn.classList.remove('hidden');
        } catch (error) {
            outputText.textContent = `Error: ${error.message}`;
            console.error('Summarization error:', error);
        } finally {
            // Hide loading spinner and show output
            loading.classList.add('hidden');
            outputText.classList.remove('hidden');
        }
    });

    // Copy summary to clipboard
    copyBtn.addEventListener('click', () => {
        const summaryText = outputText.textContent;
        navigator.clipboard.writeText(summaryText)
            .then(() => {
                const originalText = copyBtn.innerHTML;
                copyBtn.innerHTML = 'Copied! <i class="fas fa-check"></i>';
                
                setTimeout(() => {
                    copyBtn.innerHTML = originalText;
                }, 2000);
            })
            .catch(err => {
                console.error('Failed to copy text: ', err);
                alert('Failed to copy text to clipboard');
            });
    });

    // Function to generate summary using TensorFlow.js
    async function generateSummary(text, summaryRatio = 0.3) {
        // Split text into sentences
        const sentences = splitIntoSentences(text);
        
        if (sentences.length <= 3) {
            return text; // Return original text if it's too short
        }

        // Get sentence embeddings using Universal Sentence Encoder
        const embeddings = await model.embed(sentences);
        
        // Convert to 2D array
        const embeddingsArray = await embeddings.array();
        
        // Calculate similarity matrix
        const similarityMatrix = calculateSimilarityMatrix(embeddingsArray);
        
        // Calculate sentence scores using TextRank algorithm
        const scores = textRank(similarityMatrix, 0.85, 30);
        
        // Determine how many sentences to include in the summary
        const numSentencesToInclude = Math.max(3, Math.ceil(sentences.length * summaryRatio));
        
        // Get indices of top sentences
        const topIndices = getTopIndices(scores, numSentencesToInclude);
        
        // Sort indices to maintain original order
        topIndices.sort((a, b) => a - b);
        
        // Construct summary from top sentences
        const summary = topIndices.map(index => sentences[index]).join(' ');
        
        return summary;
    }

    // Helper function to split text into sentences
    function splitIntoSentences(text) {
        // Basic sentence splitting - can be improved
        const sentenceRegex = /[.!?]+\s+/g;
        const sentences = text.split(sentenceRegex).filter(s => s.trim().length > 0);
        
        // Handle case where the regex doesn't match (e.g., only one sentence)
        if (sentences.length === 0) {
            return [text];
        }
        
        return sentences;
    }

    // Calculate cosine similarity between two vectors
    function cosineSimilarity(vecA, vecB) {
        let dotProduct = 0;
        let normA = 0;
        let normB = 0;
        
        for (let i = 0; i < vecA.length; i++) {
            dotProduct += vecA[i] * vecB[i];
            normA += vecA[i] * vecA[i];
            normB += vecB[i] * vecB[i];
        }
        
        normA = Math.sqrt(normA);
        normB = Math.sqrt(normB);
        
        if (normA === 0 || normB === 0) {
            return 0;
        }
        
        return dotProduct / (normA * normB);
    }

    // Calculate similarity matrix for all sentences
    function calculateSimilarityMatrix(embeddings) {
        const numSentences = embeddings.length;
        const similarityMatrix = Array(numSentences).fill().map(() => Array(numSentences).fill(0));
        
        for (let i = 0; i < numSentences; i++) {
            for (let j = 0; j < numSentences; j++) {
                if (i === j) {
                    similarityMatrix[i][j] = 1; // Same sentence has perfect similarity
                } else {
                    const similarity = cosineSimilarity(embeddings[i], embeddings[j]);
                    similarityMatrix[i][j] = similarity;
                }
            }
        }
        
        return similarityMatrix;
    }

    // TextRank algorithm implementation
    function textRank(similarityMatrix, dampingFactor = 0.85, iterations = 30) {
        const numSentences = similarityMatrix.length;
        
        // Initialize scores
        let scores = Array(numSentences).fill(1 / numSentences);
        
        // Iterate to convergence
        for (let iter = 0; iter < iterations; iter++) {
            const newScores = Array(numSentences).fill(0);
            
            for (let i = 0; i < numSentences; i++) {
                // Random jump probability
                newScores[i] = (1 - dampingFactor) / numSentences;
                
                // Add contributions from other sentences
                for (let j = 0; j < numSentences; j++) {
                    if (i !== j) {
                        // Get outgoing link sum for sentence j
                        let outgoingSum = 0;
                        for (let k = 0; k < numSentences; k++) {
                            if (j !== k) {
                                outgoingSum += similarityMatrix[j][k];
                            }
                        }
                        
                        // Add contribution if there are outgoing links
                        if (outgoingSum > 0) {
                            newScores[i] += dampingFactor * scores[j] * similarityMatrix[j][i] / outgoingSum;
                        }
                    }
                }
            }
            
            // Update scores
            scores = newScores;
        }
        
        return scores;
    }

    // Get indices of top N scores
    function getTopIndices(scores, n) {
        return scores
            .map((score, index) => ({ score, index }))
            .sort((a, b) => b.score - a.score)
            .slice(0, n)
            .map(item => item.index);
    }

    // Function to preprocess text (remove extra whitespace, normalize, etc.)
    function preprocessText(text) {
        // Remove extra whitespace
        text = text.replace(/\s+/g, ' ').trim();
        
        // Replace common Unicode quotes with ASCII ones
        text = text.replace(/[\u2018\u2019]/g, "'");
        text = text.replace(/[\u201C\u201D]/g, '"');
        
        return text;
    }
});
