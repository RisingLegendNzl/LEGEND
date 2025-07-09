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
let simulationHelpers = {}; // This will hold the parsed helper functions from the main thread
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
 * This function replicates the core logic from the main app script,
 * but uses the provided 'individual's parameters for simulation.
 */
function calculateFitness(individual) {
    // Reconstruct the config objects for the simulation using the individual's parameters
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
    const SIM_ADAPTIVE_LEARNING_RATES = {
        SUCCESS: individual.adaptiveSuccessRate,
        FAILURE: individual.adaptiveFailureRate,
        MIN_INFLUENCE: individual.minAdaptiveInfluence,
        MAX_INFLUENCE: individual.maxAdaptiveInfluence,
    };

    // Assumed toggle states for optimization simulation (these are not optimized)
    const SIM_TOGGLES = {
        useTrendConfirmation: true, // Generally, optimizer needs to see performance with this on
        useWeightedZone: true,
        useProximityBoost: true,
        useDynamicTerminalNeighbourCount: true, // Should be part of optimization if it impacts outcome
        useNeighbourFocus: true,
        isAiReady: false, // AI prediction is not part of the fitness calculation itself for core parameters
        // Add other toggles that impact core logic, if any, and set to true/false as desired for optimization context
    };


    let wins = 0;
    let losses = 0;
    let simulatedHistory = []; // Local history for this simulation run

    // Reset temporary states for a clean simulation run
    let tempAdaptiveInfluences = { // Start fresh for each simulation
        'Hit Rate': 1.0, 'Streak': 1.0, 'Proximity to Last Spin': 1.0,
        'Hot Zone Weighting': 1.0, 'High AI Confidence': 1.0, 'Statistical Trends': 1.0
    };
    let tempConfirmedWinsLog = []; // Local confirmed wins log for this simulation run

    const sortedHistory = [...historyData].sort((a, b) => a.id - b.id); // Ensure chronological order

    for (const rawItem of sortedHistory) {
        // Critical: Check isRunning frequently within the fitness calculation loop
        if (!isRunning) {
            return 0; // Return 0 or a very low fitness if stopped mid-calculation
        }

        if (rawItem.winningNumber === null) continue; // Skip incomplete history items

        // Simulate the state *before* this spin happened using the current simulation's history
        const historyForCalc = simulatedHistory.slice();

        // Call helper functions, explicitly passing *all* necessary arguments, including configs from the individual
        const trendStats = simulationHelpers.calculateTrendStats(historyForCalc, SIM_STRATEGY_CONFIG, simulationHelpers.allPredictionTypes);
        const boardStats = simulationHelpers.getBoardStateStats(historyForCalc, SIM_STRATEGY_CONFIG);
        const neighbourScores = simulationHelpers.runNeighbourAnalysis(false, historyForCalc, SIM_STRATEGY_CONFIG, SIM_TOGGLES.useDynamicTerminalNeighbourCount);

        const recommendation = simulationHelpers.getRecommendation(
            trendStats, boardStats, neighbourScores,
            rawItem.difference, rawItem.num1, rawItem.num2,
            false, // isForWeightUpdate - false for fitness calculation
            null,  // aiPredictionData - null for core parameter optimization (AI is separate)
            tempAdaptiveInfluences, // Pass current simulation's adaptive influences
            tempConfirmedWinsLog.length > 0 ? tempConfirmedWinsLog[tempConfirmedWinsLog.length - 1] : null, // Last winning number for simulation
            SIM_TOGGLES.useProximityBoost, SIM_TOGGLES.useWeightedZone, SIM_TOGGLES.useNeighbourFocus,
            SIM_TOGGLES.isAiReady, SIM_TOGGLES.useTrendConfirmation,
            SIM_STRATEGY_CONFIG, SIM_ADAPTIVE_LEARNING_RATES,
            simulatedHistory // Pass the local simulation history for trend confirmation logic
        );

        const simItem = { ...rawItem }; // Create a copy to modify for simulation results
        simItem.recommendedGroupId = recommendation.bestCandidate ? recommendation.bestCandidate.type.id : null;
        simItem.recommendationDetails = recommendation.bestCandidate ? recommendation.bestCandidate.details : null;
        
        // Evaluate calculation status for the current simItem based on the winning number
        simulationHelpers.evaluateCalculationStatus(simItem, rawItem.winningNumber, SIM_STRATEGY_CONFIG, SIM_TOGGLES.useDynamicTerminalNeighbourCount);


        // Check if the recommendation was a win or loss
        if (simItem.recommendedGroupId) {
            if (simItem.hitTypes.includes(simItem.recommendedGroupId)) {
                wins++;
            } else {
                losses++;
            }
        }

        // Update adaptive influences based on the outcome of this simulated spin
        if (simItem.recommendedGroupId && simItem.recommendationDetails?.primaryDrivingFactor) {
             const primaryFactor = simItem.recommendationDetails.primaryDrivingFactor;
             if (tempAdaptiveInfluences[primaryFactor] === undefined) {
                 tempAdaptiveInfluences[primaryFactor] = 1.0;
             }
             const wasSuccess = simItem.hitTypes.includes(simItem.recommendedGroupId);
             if (wasSuccess) {
                tempAdaptiveInfluences[primaryFactor] = Math.min(SIM_ADAPTIVE_LEARNING_RATES.MAX_INFLUENCE, tempAdaptiveInfluences[primaryFactor] + SIM_ADAPTIVE_LEARNING_RATES.SUCCESS);
             } else {
                tempAdaptiveInfluences[primaryFactor] = Math.max(SIM_ADAPTIVE_LEARNING_RATES.MIN_INFLUENCE, tempAdaptiveInfluences[primaryFactor] - SIM_ADAPTIVE_LEARNING_RATES.FAILURE);
             }
        }
        
        simulatedHistory.push(simItem);
        if (simItem.winningNumber !== null) {
            tempConfirmedWinsLog.push(simItem.winningNumber);
        }
    }

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

            // Define a helper to extract function body
            const getFunctionBody = (funcString) => {
                const body = funcString.substring(funcString.indexOf('{') + 1, funcString.lastIndexOf('}'));
                return body.trim();
            };

            // Reconstruct helper functions with explicit parameters for their dependencies
            // This is crucial for them to work correctly within the worker's scope
            simulationHelpers = {
                // getNeighbours(number, count, rouletteWheelArr)
                getNeighbours: new Function('number', 'count', 'rouletteWheelArr',
                    getFunctionBody(payload.helpers.getNeighbours)),

                // getHitZone(baseNumber, terminals, winningNumber = null, useDynamicTerminalNeighbourCountBool, rouletteWheelArr, terminalMappingObj, getNeighboursFunc)
                getHitZone: new Function('baseNumber', 'terminals', 'winningNumber', 'useDynamicTerminalNeighbourCountBool', 'rouletteWheelArr', 'terminalMappingObj', 'getNeighboursFunc',
                    getFunctionBody(payload.helpers.getHitZone)),

                // calculatePocketDistance(num1, num2, rouletteWheelArr)
                calculatePocketDistance: new Function('num1', 'num2', 'rouletteWheelArr',
                    getFunctionBody(payload.helpers.calculatePocketDistance)),

                // calculateTrendStats(currentHistory, current_STRATEGY_CONFIG, activeTypes)
                calculateTrendStats: new Function('currentHistory', 'current_STRATEGY_CONFIG', 'activeTypes',
                    `
                    const allPredictionTypes = this.allPredictionTypes; // Access from 'this' (bound context)
                    const rouletteWheel = this.rouletteWheel; // Access from 'this' for consistency (though not used in trend stats)
                    const terminalMapping = this.terminalMapping; // Access from 'this'
                    const getNeighbours = this.getNeighbours; // Access from 'this'
                    const getHitZone = this.getHitZone; // Access from 'this'
                    const calculatePocketDistance = this.calculatePocketDistance; // Access from 'this'
                    const evaluateCalculationStatus = this.evaluateCalculationStatus; // Access from 'this'

                    ${getFunctionBody(payload.helpers.calculateTrendStats)}
                    `).bind(payload.helpers), // Bind payload.helpers to 'this' for internal access

                // getBoardStateStats(simulatedHistory, current_STRATEGY_CONFIG)
                getBoardStateStats: new Function('simulatedHistory', 'current_STRATEGY_CONFIG',
                    `
                    const allPredictionTypes = this.allPredictionTypes;
                    const rouletteWheel = this.rouletteWheel; // Access from 'this'
                    const terminalMapping = this.terminalMapping; // Access from 'this'
                    const getNeighbours = this.getNeighbours; // Access from 'this'
                    const getHitZone = this.getHitZone; // Access from 'this'
                    const calculatePocketDistance = this.calculatePocketDistance; // Access from 'this'
                    const evaluateCalculationStatus = this.evaluateCalculationStatus; // Access from 'this'
                    
                    ${getFunctionBody(payload.helpers.getBoardStateStats)}
                    `).bind(payload.helpers),

                // runNeighbourAnalysis(render, simulatedHistory, current_STRATEGY_CONFIG, current_useDynamicTerminalNeighbourCount)
                runNeighbourAnalysis: new Function('render', 'simulatedHistory', 'current_STRATEGY_CONFIG', 'current_useDynamicTerminalNeighbourCount',
                    `
                    const allPredictionTypes = this.allPredictionTypes;
                    const terminalMapping = this.terminalMapping;
                    const rouletteWheel = this.rouletteWheel;
                    const getNeighbours = this.getNeighbours; // Pass getNeighbours
                    const getHitZone = this.getHitZone; // Pass getHitZone
                    const calculatePocketDistance = this.calculatePocketDistance; // Pass calculatePocketDistance
                    const evaluateCalculationStatus = this.evaluateCalculationStatus; // Access from 'this'

                    ${getFunctionBody(payload.helpers.runNeighbourAnalysis)}
                    `).bind(payload.helpers),

                // evaluateCalculationStatus(historyItem, winningNumber, current_STRATEGY_CONFIG, useDynamicTerminalNeighbourCountBool)
                evaluateCalculationStatus: new Function('historyItem', 'winningNumber', 'current_STRATEGY_CONFIG', 'useDynamicTerminalNeighbourCountBool',
                    `
                    const allPredictionTypes = this.allPredictionTypes;
                    const terminalMapping = this.terminalMapping;
                    const rouletteWheel = this.rouletteWheel;
                    const getHitZone = this.getHitZone; // Pass getHitZone
                    const getNeighbours = this.getNeighbours; // Pass getNeighbours
                    const calculatePocketDistance = this.calculatePocketDistance; // Pass calculatePocketDistance
                    const DEBUG_MODE = false;

                    ${getFunctionBody(payload.helpers.evaluateCalculationStatus)}
                    `).bind(payload.helpers),

                // getRecommendation(...) - takes many args, ensure all are passed.
                getRecommendation: new Function(
                    'trendStats', 'boardStats', 'neighbourScores', 'diff',
                    'inputNum1', 'inputNum2', 'isForWeightUpdate', 'aiPredictionData',
                    'currentAdaptiveInfluences', 'lastWinningNumber',
                    'useProximityBoostBool', 'useWeightedZoneBool', 'useNeighbourFocusBool',
                    'isAiReadyBool', 'useTrendConfirmationBool',
                    'current_STRATEGY_CONFIG', 'current_ADAPTIVE_LEARNING_RATES', 'currentHistoryForTrend',
                    `
                    const allPredictionTypes = this.allPredictionTypes;
                    const terminalMapping = this.terminalMapping;
                    const rouletteWheel = this.rouletteWheel;
                    const getHitZone = this.getHitZone; // Access getHitZone from 'this'
                    const getNeighbours = this.getNeighbours; // Access getNeighbours from 'this'
                    const calculatePocketDistance = this.calculatePocketDistance; // Access calculatePocketDistance from 'this'
                    const useDynamicTerminalNeighbourCount = this.useDynamicTerminalNeighbourCount; // Fixed toggle value from main thread
                    const DEBUG_MODE = false; // Always false in worker for performance
                    
                    ${getFunctionBody(payload.helpers.getRecommendation)}
                    `).bind(payload.helpers),
            };
            
            runEvolution();
            break;

        case 'stop':
            isRunning = false; // Set isRunning to false to stop the loop
            break;
    }
};
