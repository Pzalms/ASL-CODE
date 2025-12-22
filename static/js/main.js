/**
 * ASL Recognition System - Main JavaScript
 * Handles real-time predictions, user interactions, and audio playback
 */

// Constants
const ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
const SPECIAL_KEYS = ['del', 'nothing', 'space'];
const ALL_LABELS = [...ALPHABET, ...SPECIAL_KEYS];
const PREDICTION_INTERVAL = 500; // ms

// DOM Elements
const elements = {
    videoFeed: document.getElementById('videoFeed'),
    predictionLetter: document.getElementById('predictionLetter'),
    predictionConfidence: document.getElementById('predictionConfidence'),
    wordDisplay: document.getElementById('wordDisplay'),
    modelStatus: document.getElementById('modelStatus'),
    alphabetGrid: document.getElementById('alphabetGrid'),
    addLetterBtn: document.getElementById('addLetterBtn'),
    speakBtn: document.getElementById('speakBtn'),
    clearBtn: document.getElementById('clearBtn'),
    audioPlayer: document.getElementById('audioPlayer')
};

// State
let currentPrediction = { label: '', confidence: 0 };
let currentWord = '';
let predictionInterval = null;

/**
 * Initialize the application
 */
function init() {
    createAlphabetGrid();
    setupEventListeners();
    checkModelStatus();
    startPredictionPolling();
}

/**
 * Create the alphabet reference grid
 */
function createAlphabetGrid() {
    const grid = elements.alphabetGrid;
    grid.innerHTML = '';

    ALL_LABELS.forEach(letter => {
        const tile = document.createElement('div');
        tile.className = 'letter-tile';
        tile.dataset.letter = letter;
        tile.textContent = letter === 'nothing' ? '∅' :
            letter === 'space' ? '␣' :
                letter === 'del' ? '⌫' : letter;
        grid.appendChild(tile);
    });
}

/**
 * Set up event listeners
 */
function setupEventListeners() {
    elements.addLetterBtn.addEventListener('click', addLetter);
    elements.speakBtn.addEventListener('click', speakWord);
    elements.clearBtn.addEventListener('click', clearWord);

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.code === 'Space' && !e.target.matches('input, textarea')) {
            e.preventDefault();
            addLetter();
        } else if (e.code === 'Enter') {
            speakWord();
        } else if (e.code === 'Escape') {
            clearWord();
        }
    });
}

/**
 * Check if the model is loaded
 */
async function checkModelStatus() {
    try {
        const response = await fetch('/model_status');
        const data = await response.json();

        const statusEl = elements.modelStatus;
        if (data.loaded) {
            statusEl.classList.add('ready');
            statusEl.classList.remove('error');
            statusEl.querySelector('.status-text').textContent = 'Model Ready';
        } else {
            statusEl.classList.remove('ready');
            statusEl.classList.add('error');
            statusEl.querySelector('.status-text').textContent = 'Model Not Loaded';
        }
    } catch (error) {
        console.error('Error checking model status:', error);
        elements.modelStatus.classList.add('error');
        elements.modelStatus.querySelector('.status-text').textContent = 'Connection Error';
    }
}

/**
 * Start polling for predictions
 */
function startPredictionPolling() {
    if (predictionInterval) {
        clearInterval(predictionInterval);
    }

    predictionInterval = setInterval(fetchPrediction, PREDICTION_INTERVAL);
}

/**
 * Fetch the current prediction from the server
 */
async function fetchPrediction() {
    try {
        const response = await fetch('/predict');
        const data = await response.json();

        currentPrediction = data;
        updatePredictionDisplay(data);
    } catch (error) {
        console.error('Error fetching prediction:', error);
    }
}

/**
 * Update the prediction display
 */
function updatePredictionDisplay(prediction) {
    const { label, confidence } = prediction;

    // Update letter display
    elements.predictionLetter.textContent = label || '-';

    // Update confidence
    const confPercent = Math.round((confidence || 0) * 100);
    elements.predictionConfidence.textContent = `${confPercent}%`;

    // Highlight the detected letter in the grid
    document.querySelectorAll('.letter-tile').forEach(tile => {
        tile.classList.toggle('active', tile.dataset.letter === label);
    });
}

/**
 * Add the current letter to the word
 */
async function addLetter() {
    try {
        const response = await fetch('/add_letter', { method: 'POST' });
        const data = await response.json();

        currentWord = data.word;
        updateWordDisplay();

        // Visual feedback
        elements.addLetterBtn.style.transform = 'scale(0.95)';
        setTimeout(() => {
            elements.addLetterBtn.style.transform = '';
        }, 100);
    } catch (error) {
        console.error('Error adding letter:', error);
    }
}

/**
 * Clear the word buffer
 */
async function clearWord() {
    try {
        await fetch('/clear_word', { method: 'POST' });
        currentWord = '';
        updateWordDisplay();
    } catch (error) {
        console.error('Error clearing word:', error);
    }
}

/**
 * Speak the current word
 */
async function speakWord() {
    if (!currentWord.trim()) {
        return;
    }

    try {
        elements.speakBtn.disabled = true;
        elements.speakBtn.textContent = 'Speaking...';

        const response = await fetch('/speak', { method: 'POST' });
        const data = await response.json();

        if (data.audio) {
            const audioData = `data:${data.content_type};base64,${data.audio}`;
            elements.audioPlayer.src = audioData;
            elements.audioPlayer.play();
        } else if (data.error) {
            console.error('Speech error:', data.error);
        }
    } catch (error) {
        console.error('Error speaking:', error);
    } finally {
        elements.speakBtn.disabled = false;
        elements.speakBtn.innerHTML = '<span class="btn-icon">🔊</span>Speak';
    }
}

/**
 * Update the word display
 */
function updateWordDisplay() {
    if (currentWord) {
        elements.wordDisplay.textContent = currentWord;
        elements.wordDisplay.classList.remove('placeholder');
    } else {
        elements.wordDisplay.innerHTML = '<span class="placeholder-text">Show signs to camera...</span>';
    }
}

/**
 * Fetch the current word from the server
 */
async function syncWord() {
    try {
        const response = await fetch('/get_word');
        const data = await response.json();
        currentWord = data.word;
        updateWordDisplay();
    } catch (error) {
        console.error('Error syncing word:', error);
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', init);
