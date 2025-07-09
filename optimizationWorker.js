// optimizationWorker.js - Genetic Algorithm for Parameter Optimization

// --- GENETIC ALGORITHM CONFIGURATION ---
const GA_CONFIG = {
    populationSize: 50,    // Number of parameter sets in each generation
    mutationRate: 0.15,    // Probability of a parameter being randomly changed
    crossoverRate: 0.7,    // Probability of two parents breeding
    eliteCount: 4,         // Number of top performers to carry over to the next generation
    maxGenerations: 100     // Maximum number of generations to run
};

// --- PARAMETER DEFINITIONS (The "Genes") ---
// Defines the boundaries and step for each parameter to be optimized.
const parameterSpace = {
    learningRate_success: { min: 0.01, max: 1.0, step: 0.01 },
    learningRate_failure: { min: 0.01, max: 0.5, step: 0.01 },
    maxWeight:            { min: 1.0, max: 10.0, step: 0.1 },
    minWeight:            { min: 0.0, max: 1.0, step: 0.01 },
    decayFactor:          { min: 0.7, max: 0.99, step: 0.01 },
    patternMinAttempts:   { min: 1, max: 20, step: 1 },
    patternSuccessThreshold: { min: 50, max: 100, step: 1 },
    triggerMinAttempts:   { min: 1, max: 20, step: 1 },
    triggerSuccessThreshold: { min: 50, max: 100, step: 1 },
    adaptiveSuccessRate:  { min: 0.01, max: 0.5, step: 0.01 },
    adaptiveFailureRate:  { min: 0.01, max: 0.5, step: 0.01 },
    minAdaptiveInfluence: { min: 0.0, max: 1.0, step: 0.01 },
    maxAdaptiveInfluence: { min: 1.0, max: 5.0, step: 0.1 }
};

let historyData = [];
let simulationHelpers = {};
let isRunning = false;
let generationCount = 0;

// --- CORE GENETIC ALGORITHM LOGIC ---

// Creates a single random individual (a full set of parameters)
function createIndividual() {
    const individual = {};
    for (const key in parameterSpace) {
        const { min, max, step } = parameterSpace[key];
        const range = (max - min) / step;
        const randomStep = Math.floor(Math.random() * (range + 1));
        individual[key] = parseFloat((min + randomStep * step).toFixed(4));
    }
    return individual;
}

// Breeds two parents to create a child
function crossover(parent1, parent2) {
    const child = {};
    for (const key in parent1) {
        // Uniform crossover: 50/50 chance to inherit a gene from either parent
        child[key] = Math.random() < 0.5 ? parent1[key] : parent2[key];
    }
    return child;
}

// Randomly mutates an individual's genes
function mutate(individual) {
    for (const key in individual) {
        if (Math.random() < GA_CONFIG.mutationRate) {
            const { min, max, step } = parameterSpace[key];
            const range = (max - min) / step;
            const randomStep = Math.floor(Math.random() * (range + 1));
            individual[key] = parseFloat((min + randomStep * step).toFixed(4));
        }
    }
    return individual;
}

// Selects parents from the population based on fitness (Tournament Selection)
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

/**
 * This is the most critical function. It takes a set of parameters (an individual)
 * and runs a full simulation on the historical data to calculate a win/loss ratio.
 * This function replicates the core logic from the main app script.
 */
function calculateFitness(individual) {
    // Reconstruct the config objects for the simulation
    const STRATEGY_CONFIG = {
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
    const ADAPTIVE_LEARNING_RATES = {
        SUCCESS: individual.adaptiveSuccessRate,
        FAILURE: individual.adaptiveFailureRate,
        MIN_INFLUENCE: individual.minAdaptiveInfluence,
        MAX_INFLUENCE: individual.maxAdaptiveInfluence,
    };

    // --- Start of Simulation Logic (Adapted from main script) ---
    // Note: Toggles are assumed to be enabled for full strategy evaluation
    let wins = 0;
    let losses = 0;
    let simulatedHistory = [];

    // Reset temporary states for a clean simulation run
    let tempStrategyStates = {
        weightedZone: { weight: 1.0 },
        proximityBoost: { weight: 1.0 }
    };
    let tempAdaptiveInfluences = {
        'Hit Rate': 1.0, 'Streak': 1.0, 'Proximity to Last Spin': 1.0,
        'Hot Zone Weighting': 1.0, 'High AI Confidence': 1.0, 'Statistical Trends': 1.0
    };

    const sortedHistory = [...historyData].sort((a, b) => a.id - b.id);
    let tempConfirmedWinsLog = [];

    for (const rawItem of sortedHistory) {
         // ADDED: Check isRunning frequently within the fitness calculation loop
        if (!isRunning) return 0; // Return 0 or a very low fitness if stopped mid-calculation

         if (rawItem.winningNumber === null) continue;

        // Simulate the state *before* this spin happened
        const historyForCalc = simulatedHistory.slice();
        // Pass STRATEGY_CONFIG to helper functions that use it
        const trendStats = simulationHelpers.calculateTrendStats(historyForCalc, simulationHelpers.allPredictionTypes, STRATEGY_CONFIG.decayFactor); // Modified here
        const boardStats = simulationHelpers.getBoardStateStats(historyForCalc, STRATEGY_CONFIG.decayFactor); // Modified here
        const neighbourScores = simulationHelpers.runNeighbourAnalysis(false, historyForCalc, STRATEGY_CONFIG.decayFactor); // Modified here

        // Also pass STRATEGY_CONFIG and ADAPTIVE_LEARNING_RATES to getRecommendation as it relies on global STRATEGY_CONFIG
        const recommendation = simulationHelpers.getRecommendation(
            trendStats, boardStats, neighbourScores,
            rawItem.difference, rawItem.num1, rawItem.num2,
            false, null, tempAdaptiveInfluences,
            tempConfirmedWinsLog.length > 0 ? tempConfirmedWinsLog[tempConfirmedWinsLog.length - 1] : null,
            STRATEGY_CONFIG, // Pass STRATEGY_CONFIG
            ADAPTIVE_LEARNING_RATES // Pass ADAPTIVE_LEARNING_RATES
        );

        const simItem = { ...rawItem };
        simItem.recommendedGroupId = recommendation.bestCandidate ? recommendation.bestCandidate.type.id : null;
        simItem.recommendationDetails = recommendation.bestCandidate ? recommendation.bestCandidate.details : null;

        // Check if the recommendation was a win or loss
        if (simItem.recommendedGroupId) {
            if (simItem.hitTypes.includes(simItem.recommendedGroupId)) {
                wins++;
            } else {
                losses++;
            }
        }

        // Update strategy weights and adaptive influences based on the outcome of this simulated spin
        if (simItem.recommendedGroupId) {
            const wasSuccess = simItem.hitTypes.includes(simItem.recommendedGroupId);

            // Update strategy weights
            // Ensure tempStrategyStates exists
            if (!tempStrategyStates.weightedZone) tempStrategyStates.weightedZone = { weight: 1.0 };
            if (!tempStrategyStates.proximityBoost) tempStrategyStates.proximityBoost = { weight: 1.0 };

            // These updates should be based on the individual's parameters (STRATEGY_CONFIG derived from 'individual')
            tempStrategyStates.weightedZone.weight = wasSuccess
                ? Math.min(STRATEGY_CONFIG.maxWeight, tempStrategyStates.weightedZone.weight + STRATEGY_CONFIG.learningRate_success)
                : Math.max(STRATEGY_CONFIG.minWeight, tempStrategyStates.weightedZone.weight - STRATEGY_CONFIG.learningRate_failure);
            
            tempStrategyStates.proximityBoost.weight = wasSuccess
                ? Math.min(STRATEGY_CONFIG.maxWeight, tempStrategyStates.proximityBoost.weight + STRATEGY_CONFIG.learningRate_success)
                : Math.max(STRATEGY_CONFIG.minWeight, tempStrategyStates.proximityBoost.weight - STRATEGY_CONFIG.learningRate_failure);


            // Update adaptive influences
            if (simItem.recommendationDetails?.primaryDrivingFactor) {
                 const primaryFactor = simItem.recommendationDetails.primaryDrivingFactor;
                 // Ensure tempAdaptiveInfluences has the factor initialized
                 if (tempAdaptiveInfluences[primaryFactor] === undefined) {
                     tempAdaptiveInfluences[primaryFactor] = 1.0;
                 }
                 if (wasSuccess) {
                    tempAdaptiveInfluences[primaryFactor] = Math.min(ADAPTIVE_LEARNING_RATES.MAX_INFLUENCE, tempAdaptiveInfluences[primaryFactor] + ADAPTIVE_LEARNING_RATES.SUCCESS);
                 } else {
                    tempAdaptiveInfluences[primaryFactor] = Math.max(ADAPTIVE_LEARNING_RATES.MIN_INFLUENCE, tempAdaptiveInfluences[primaryFactor] - ADAPTIVE_LEARNING_RATES.FAILURE);
                 }
            }
        }
        
        simulatedHistory.push(simItem);
        if (simItem.winningNumber !== null) {
            tempConfirmedWinsLog.push(simItem.winningNumber);
        }
    }
    // --- End of Simulation Logic ---

    // Fitness is the W/L ratio. Handle division by zero.
    if (losses === 0) {
        return wins > 0 ? wins * 10 : 0; // Heavily reward perfect records
    }
    return wins / losses;
}

// --- MAIN EVOLUTION LOOP ---

async function runEvolution() {
    isRunning = true;
    generationCount = 0;

    // 1. Create initial population
    let population = [];
    for (let i = 0; i < GA_CONFIG.populationSize; i++) {
        population.push({ individual: createIndividual(), fitness: 0 });
    }

    // Main loop for generations
    while (isRunning && generationCount < GA_CONFIG.maxGenerations) {
        generationCount++;

        // 2. Calculate fitness for each individual
        for (const p of population) {
            if (!isRunning) { // Check for stop command before calculating each fitness
                self.postMessage({ type: 'stopped' }); // Inform main thread that we stopped
                return; // Exit the function immediately
            }
            p.fitness = calculateFitness(p.individual);
        }

        if (!isRunning) { // Redundant check, but harmless if the inner loop return is missed
            self.postMessage({ type: 'stopped' });
            return;
        }

        // Sort by fitness (descending)
        population.sort((a, b) => b.fitness - a.fitness);

        // Post progress back to the main thread
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

        // 3. Elitism: Keep the best individuals
        for (let i = 0; i < GA_CONFIG.eliteCount; i++) {
            newPopulation.push(population[i]);
        }

        // 4. Crossover & Mutation
        while (newPopulation.length < GA_CONFIG.populationSize) {
            if (!isRunning) { // Check for stop command during population generation
                self.postMessage({ type: 'stopped' }); // Inform main thread that we stopped
                return; // Exit the function immediately
            }
            const parent1 = selectParent(population);
            const parent2 = selectParent(population);
            let child;

            if (Math.random() < GA_CONFIG.crossoverRate) {
                child = crossover(parent1.individual, parent2.individual);
            } else {
                child = { ...parent1.individual }; // Cloning
            }

            child = mutate(child);
            newPopulation.push({ individual: child, fitness: 0 });
        }
        
        population = newPopulation;
    }

    // If loop finishes naturally
    if (isRunning) { // This condition checks if it finished due to maxGenerations, not a stop command
        self.postMessage({
            type: 'complete',
            payload: {
                generation: generationCount,
                bestFitness: population[0].fitness.toFixed(3),
                bestIndividual: population[0].individual
            }
        });
    }
    isRunning = false; // Ensure isRunning is false at the end
}

// --- WEB WORKER MESSAGE HANDLER ---
self.onmessage = (event) => {
    const { type, payload } = event.data;

    switch (type) {
        case 'start':
            if (isRunning) return;
            historyData = payload.history;
            // The main thread sends us the helper functions it uses for simulation
            // We must deserialize them from strings.
            simulationHelpers = {
                // Modified helper function deserialization to pass STRATEGY_CONFIG and ADAPTIVE_LEARNING_RATES
                // where needed, as these are now dynamic based on the individual.
                // The new Function constructor is used to re-create the functions from their string representations.
                calculateTrendStats: new Function('currentHistory', 'activeTypes', 'decayFactor', 'STRATEGY_CONFIG', 'return ' + payload.helpers.calculateTrendStats.substring(payload.helpers.calculateTrendStats.indexOf('{') + 1, payload.helpers.calculateTrendStats.lastIndexOf('}')))(),
                getBoardStateStats: new Function('simulatedHistory', 'decayFactor', 'STRATEGY_CONFIG', 'return ' + payload.helpers.getBoardStateStats.substring(payload.helpers.getBoardStateStats.indexOf('{') + 1, payload.helpers.getBoardStateStats.lastIndexOf('}')))(),
                getRecommendation: new Function('trendStats', 'boardStats', 'neighbourScores', 'diff', 'inputNum1', 'inputNum2', 'isForWeightUpdate', 'aiPredictionData', 'currentAdaptiveInfluences', 'lastWinningNumber', 'STRATEGY_CONFIG', 'ADAPTIVE_LEARNING_RATES', 'return ' + payload.helpers.getRecommendation.substring(payload.helpers.getRecommendation.indexOf('{') + 1, payload.helpers.getRecommendation.lastIndexOf('}')))(),
                runNeighbourAnalysis: new Function('render', 'simulatedHistory', 'decayFactor', 'STRATEGY_CONFIG', 'return ' + payload.helpers.runNeighbourAnalysis.substring(payload.helpers.runNeighbourAnalysis.indexOf('{') + 1, payload.helpers.runNeighbourAnalysis.lastIndexOf('}')))(),
                getHitZone: new Function('baseNumber', 'terminals', 'winningNumber', 'useDynamicTerminalNeighbourCount', 'rouletteWheel', 'getNeighbours', 'return ' + payload.helpers.getHitZone.substring(payload.helpers.getHitZone.indexOf('{') + 1, payload.helpers.getHitZone.lastIndexOf('}')))(),
                getNeighbours: new Function('number', 'count', 'rouletteWheel', 'return ' + payload.helpers.getNeighbours.substring(payload.helpers.getNeighbours.indexOf('{') + 1, payload.helpers.getNeighbours.lastIndexOf('}')))(),
                calculatePocketDistance: new Function('num1', 'num2', 'rouletteWheel', 'return ' + payload.helpers.calculatePocketDistance.substring(payload.helpers.calculatePocketDistance.indexOf('{') + 1, payload.helpers.calculatePocketDistance.lastIndexOf('}')))(),
                
                allPredictionTypes: payload.helpers.allPredictionTypes, 
                terminalMapping: payload.helpers.terminalMapping,
                rouletteWheel: payload.helpers.rouletteWheel,
            };

            // Bind constants to the helper functions after they are deserialized
            // This ensures they have access to the necessary data from the worker's scope
            simulationHelpers.calculateTrendStats = (currentHistory, activeTypes, decayFactor) => {
                const originalFunction = new Function('STRATEGY_CONFIG', 'allPredictionTypes', 'return ' + payload.helpers.calculateTrendStats)();
                return originalFunction.call(null, STRATEGY_CONFIG, activeTypes || simulationHelpers.allPredictionTypes); // Pass STRATEGY_CONFIG
            };

            simulationHelpers.getBoardStateStats = (simulatedHistory, decayFactor) => {
                const originalFunction = new Function('STRATEGY_CONFIG', 'allPredictionTypes', 'return ' + payload.helpers.getBoardStateStats)();
                return originalFunction.call(null, simulatedHistory, STRATEGY_CONFIG); // Pass STRATEGY_CONFIG
            };
            
            simulationHelpers.runNeighbourAnalysis = (render, simulatedHistory, decayFactor) => {
                const originalFunction = new Function('STRATEGY_CONFIG', 'allPredictionTypes', 'terminalMapping', 'rouletteWheel', 'getNeighbours', 'useDynamicTerminalNeighbourCount', 'return ' + payload.helpers.runNeighbourAnalysis)();
                return originalFunction.call(null, render, simulatedHistory, STRATEGY_CONFIG, simulationHelpers.terminalMapping, simulationHelpers.rouletteWheel, simulationHelpers.getNeighbours, simulationHelpers.useDynamicTerminalNeighbourCount);
            };

            simulationHelpers.getRecommendation = (trendStats, boardStats, neighbourScores, diff, inputNum1, inputNum2, isForWeightUpdate, aiPredictionData, currentAdaptiveInfluences, lastWinningNumber, currentSTRATEGY_CONFIG, currentADAPTIVE_LEARNING_RATES) => {
                // Ensure all necessary parameters and global-like variables are passed
                const originalFunction = new Function(
                    'trendStats', 'boardStats', 'neighbourScores', 'diff', 
                    'inputNum1', 'inputNum2', 'isForWeightUpdate', 'aiPredictionData', 
                    'currentAdaptiveInfluences', 'confirmedWinsLog', 
                    'allPredictionTypes', 'terminalMapping', 'rouletteWheel', 
                    'getHitZone', 'getNeighbours', 'calculatePocketDistance',
                    'useProximityBoost', 'useWeightedZone', 'useNeighbourFocus', 'isAiReady', 'useTrendConfirmation',
                    'STRATEGY_CONFIG_PARAM', 'ADAPTIVE_LEARNING_RATES_PARAM', // Renamed to avoid clash with local const
                    'history', // Passed for trend confirmation logic
                    'return ' + payload.helpers.getRecommendation)();

                return originalFunction.call(null, 
                    trendStats, boardStats, neighbourScores, diff, 
                    inputNum1, inputNum2, isForWeightUpdate, aiPredictionData, 
                    currentAdaptiveInfluences, historyData.filter(item => item.winningNumber !== null).map(item => item.winningNumber), // Pass current confirmedWinsLog from historyData
                    simulationHelpers.allPredictionTypes, simulationHelpers.terminalMapping, simulationHelpers.rouletteWheel,
                    simulationHelpers.getHitZone, simulationHelpers.getNeighbours, simulationHelpers.calculatePocketDistance,
                    true, true, true, // Assuming these toggles are always true for optimization purposes
                    false, // Assuming trend confirmation can be false for initial optimization flexibility
                    currentSTRATEGY_CONFIG, currentADAPTIVE_LEARNING_RATES,
                    historyData // Pass history for trend confirmation logic
                );
            };

             simulationHelpers.getHitZone = (baseNumber, terminals, winningNumber, useDynamicTerminalNeighbourCount, rouletteWheel, getNeighbours) => {
                const originalFunction = new Function('terminalMapping', 'rouletteWheel', 'getNeighbours', 'useDynamicTerminalNeighbourCount', 'return ' + payload.helpers.getHitZone)();
                return originalFunction.call(null, baseNumber, terminals, winningNumber);
            };

            simulationHelpers.getNeighbours = (number, count) => {
                const originalFunction = new Function('rouletteWheel', 'return ' + payload.helpers.getNeighbours)();
                return originalFunction.call(null, number, count);
            };

            simulationHelpers.calculatePocketDistance = (num1, num2) => {
                const originalFunction = new Function('rouletteWheel', 'return ' + payload.helpers.calculatePocketDistance)();
                return originalFunction.call(null, num1, num2);
            };
            
            runEvolution();
            break;

        case 'stop':
            isRunning = false; // Set isRunning to false to stop the loop
            break;
    }
};
