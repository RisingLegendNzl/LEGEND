// optimizationWorker.js - Genetic Algorithm for Parameter Optimization

// Import the shared logic file. This is MUCH more robust than rebuilding functions from strings.
importScripts('shared-logic.js');

// --- GENETIC ALGORITHM CONFIGURATION ---
const GA_CONFIG = {
    populationSize: 50,
    mutationRate: 0.15,
    crossoverRate: 0.7,
    eliteCount: 4,
    maxGenerations: 100
};

// --- PARAMETER DEFINITIONS (The "Genes") ---
const parameterSpace = {
    learningRate_success: { min: 0.01, max: 1.0, step: 0.01 },
    learningRate_failure: { min: 0.01, max: 0.5, step: 0.01 },
    maxWeight: { min: 1.0, max: 10.0, step: 0.1 },
    minWeight: { min: 0.0, max: 1.0, step: 0.01 },
    decayFactor: { min: 0.7, max: 0.99, step: 0.01 },
    patternMinAttempts: { min: 1, max: 20, step: 1 },
    patternSuccessThreshold: { min: 50, max: 100, step: 1 },
    triggerMinAttempts: { min: 1, max: 20, step: 1 },
    triggerSuccessThreshold: { min: 50, max: 100, step: 1 },
    adaptiveSuccessRate: { min: 0.01, max: 0.5, step: 0.01 },
    adaptiveFailureRate: { min: 0.01, max: 0.5, step: 0.01 },
    minAdaptiveInfluence: { min: 0.0, max: 1.0, step: 0.01 },
    maxAdaptiveInfluence: { min: 1.0, max: 5.0, step: 0.1 }
};

let historyData = [];
let sharedData = {}; // To hold data from main thread like rouletteWheel, terminalMapping etc.
let isRunning = false;
let generationCount = 0;

// --- CORE GENETIC ALGORITHM LOGIC (Unchanged) ---
function createIndividual() {
    const individual = {};
    for (const key in parameterSpace) {
        const { min, max, step } = parameterSpace[key];
        const randomStep = Math.floor(Math.random() * (((max - min) / step) + 1));
        individual[key] = parseFloat((min + randomStep * step).toFixed(4));
    }
    return individual;
}

function crossover(parent1, parent2) {
    const child = {};
    for (const key in parent1) {
        child[key] = Math.random() < 0.5 ? parent1[key] : parent2[key];
    }
    return child;
}

function mutate(individual) {
    for (const key in individual) {
        if (Math.random() < GA_CONFIG.mutationRate) {
            const { min, max, step } = parameterSpace[key];
            const randomStep = Math.floor(Math.random() * (((max - min) / step) + 1));
            individual[key] = parseFloat((min + randomStep * step).toFixed(4));
        }
    }
    return individual;
}

function selectParent(population) {
    const tournamentSize = 5;
    let best = null;
    for (let i = 0; i < tournamentSize; i++) {
        const randomIndividual = population[Math.floor(Math.random() * population.length)];
        if (best === null || randomIndividual.fitness > best.fitness) {
            best = randomIndividual;
        }
    }
    return best;
}

// --- FITNESS CALCULATION (SIMULATION) ---
function calculateFitness(individual) {
    const SIM_STRATEGY_CONFIG = {
        learningRate_success: individual.learningRate_success,
        learningRate_failure: individual.learningRate_failure,
        maxWeight: individual.maxWeight,
        minWeight: individual.minWeight,
        decayFactor: individual.decayFactor,
        patternMinAttempts: individual.patternMinAttempts,
        patternSuccessThreshold: individual.patternSuccessThreshold,
        triggerMinAttempts: individual.triggerMinAttempts,
        triggerSuccessThreshold: individual.triggerSuccessThreshold,
    };

    let wins = 0;
    let losses = 0;
    let simulatedHistory = [];
    let tempConfirmedWinsLog = [];
    const sortedHistory = [...historyData].sort((a, b) => a.id - b.id);

    for (const rawItem of sortedHistory) {
        if (!isRunning) return 0;
        if (rawItem.winningNumber === null) continue;

        const trendStats = calculateTrendStats(simulatedHistory, SIM_STRATEGY_CONFIG, sharedData.allPredictionTypes);
        const boardStats = getBoardStateStats(simulatedHistory, SIM_STRATEGY_CONFIG, sharedData.allPredictionTypes);
        const neighbourScores = runNeighbourAnalysis(simulatedHistory, SIM_STRATEGY_CONFIG, true, sharedData.allPredictionTypes, sharedData.terminalMapping, sharedData.rouletteWheel);

        const recommendation = getRecommendation({
            trendStats, boardStats, neighbourScores,
            diff: rawItem.difference, inputNum1: rawItem.num1, inputNum2: rawItem.num2,
            currentAdaptiveInfluences: {}, lastWinningNumber: tempConfirmedWinsLog.length > 0 ? tempConfirmedWinsLog[tempConfirmedWinsLog.length - 1] : null,
            useProximityBoostBool: true, useWeightedZoneBool: true, useNeighbourFocusBool: true,
            isAiReadyBool: false, useTrendConfirmationBool: true,
            current_STRATEGY_CONFIG: SIM_STRATEGY_CONFIG,
            activePredictionTypes: sharedData.allPredictionTypes,
            allPredictionTypes: sharedData.allPredictionTypes,
            terminalMapping: sharedData.terminalMapping,
            rouletteWheel: sharedData.rouletteWheel
        });

        const simItem = { ...rawItem };
        simItem.recommendedGroupId = recommendation.bestCandidate ? recommendation.bestCandidate.type.id : null;
        
        evaluateCalculationStatus(simItem, rawItem.winningNumber, true, sharedData.allPredictionTypes, sharedData.terminalMapping, sharedData.rouletteWheel);

        if (simItem.recommendedGroupId && simItem.hitTypes.includes(simItem.recommendedGroupId)) {
            wins++;
        } else if (simItem.recommendedGroupId) {
            losses++;
        }
        
        simulatedHistory.push(simItem);
        if (simItem.winningNumber !== null) tempConfirmedWinsLog.push(simItem.winningNumber);
    }

    if (losses === 0) return wins > 0 ? wins * 10 : 0;
    return wins / losses;
}

// --- MAIN EVOLUTION LOOP (Largely Unchanged) ---
async function runEvolution() {
    isRunning = true;
    generationCount = 0;
    let population = [];
    for (let i = 0; i < GA_CONFIG.populationSize; i++) {
        population.push({ individual: createIndividual(), fitness: 0 });
    }

    while (isRunning && generationCount < GA_CONFIG.maxGenerations) {
        generationCount++;
        for (const p of population) {
            if (!isRunning) { self.postMessage({ type: 'stopped' }); return; }
            p.fitness = calculateFitness(p.individual);
        }
        if (!isRunning) { self.postMessage({ type: 'stopped' }); return; }

        population.sort((a, b) => b.fitness - a.fitness);
        self.postMessage({
            type: 'progress',
            payload: {
                generation: generationCount,
                maxGenerations: GA_CONFIG.maxGenerations,
                bestFitness: population[0].fitness.toFixed(3),
                bestIndividual: population[0].individual,
                processedCount: generationCount * GA_CONFIG.populationSize
            }
        });

        const newPopulation = [];
        for (let i = 0; i < GA_CONFIG.eliteCount; i++) newPopulation.push(population[i]);
        while (newPopulation.length < GA_CONFIG.populationSize) {
            if (!isRunning) { self.postMessage({ type: 'stopped' }); return; }
            const parent1 = selectParent(population);
            const parent2 = selectParent(population);
            let child = (Math.random() < GA_CONFIG.crossoverRate) ? crossover(parent1.individual, parent2.individual) : { ...parent1.individual };
            child = mutate(child);
            newPopulation.push({ individual: child, fitness: 0 });
        }
        population = newPopulation;
    }

    if (isRunning) {
        self.postMessage({
            type: 'complete',
            payload: {
                generation: generationCount,
                bestFitness: population[0].fitness.toFixed(3),
                bestIndividual: population[0].individual
            }
        });
    }
    isRunning = false;
}

// --- WEB WORKER MESSAGE HANDLER ---
self.onmessage = (event) => {
    const { type, payload } = event.data;
    switch (type) {
        case 'start':
            if (isRunning) return;
            historyData = payload.history;
            // Store the data needed by the shared functions
            sharedData = {
                allPredictionTypes: payload.helpers.allPredictionTypes,
                terminalMapping: payload.helpers.terminalMapping,
                rouletteWheel: payload.helpers.rouletteWheel
            };
            runEvolution();
            break;
        case 'stop':
            isRunning = false;
            break;
    }
};