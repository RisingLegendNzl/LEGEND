// shared-logic.js

// This file contains core calculation logic shared between the main app (index.html)
// and the optimization web worker (optimizationWorker.js).

/**
 * NOTE: These functions are designed to be "pure" where possible.
 * They do not access global variables from index.html directly. Instead, they
 * receive all necessary data (like the rouletteWheel, terminalMapping, configs)
 * as parameters. This makes them predictable and testable.
 */

function getNeighbours(number, count, rouletteWheel) {
    const index = rouletteWheel.indexOf(number);
    if (index === -1) return [];
    const neighbours = new Set();
    const wheelSize = rouletteWheel.length;
    for (let i = 1; i <= count; i++) {
        neighbours.add(rouletteWheel[(index - i + wheelSize) % wheelSize]);
        neighbours.add(rouletteWheel[(index + i) % wheelSize]);
    }
    return Array.from(neighbours);
}

function calculatePocketDistance(num1, num2, rouletteWheel) {
    const index1 = rouletteWheel.indexOf(num1);
    const index2 = rouletteWheel.indexOf(num2);
    if (index1 === -1 || index2 === -1) return Infinity;
    const directDistance = Math.abs(index1 - index2);
    const wrapAroundDistance = rouletteWheel.length - directDistance;
    return Math.min(directDistance, wrapAroundDistance);
}

function getHitZone(baseNumber, terminals, winningNumber, useDynamicTerminalNeighbourCountBool, terminalMapping, rouletteWheel) {
    if (baseNumber < 0 || baseNumber > 36) return [];
    const hitZone = new Set([baseNumber]);
    const numTerminals = terminals ? terminals.length : 0;

    let baseNeighbourCount = (numTerminals === 1) ? 3 : (numTerminals >= 2) ? 1 : 0;
    if (baseNeighbourCount > 0) getNeighbours(baseNumber, baseNeighbourCount, rouletteWheel).forEach(n => hitZone.add(n));

    let terminalNeighbourCount;
    if (useDynamicTerminalNeighbourCountBool && winningNumber !== null) {
        if (baseNumber === winningNumber || (terminals && terminals.includes(winningNumber))) {
            terminalNeighbourCount = 0;
        } else {
            terminalNeighbourCount = (numTerminals === 1 || numTerminals === 2) ? 3 : (numTerminals > 2) ? 1 : 0;
        }
    } else {
        terminalNeighbourCount = (numTerminals === 1 || numTerminals === 2) ? 3 : (numTerminals > 2) ? 1 : 0;
    }

    if (terminals && terminals.length > 0) {
        terminals.forEach(t => {
            hitZone.add(t);
            if (terminalNeighbourCount > 0) getNeighbours(t, terminalNeighbourCount, rouletteWheel).forEach(n => hitZone.add(n));
        });
    }
    return Array.from(hitZone);
}

function evaluateCalculationStatus(historyItem, winningNumber, useDynamicTerminalNeighbourCountBool, activePredictionTypes, terminalMapping, rouletteWheel) {
    historyItem.winningNumber = winningNumber;
    historyItem.hitTypes = [];
    historyItem.typeSuccessStatus = {};
    let minPocketDistance = Infinity;

    activePredictionTypes.forEach(type => {
        const baseNum = type.calculateBase(historyItem.num1, historyItem.num2);
        if (baseNum < 0 || baseNum > 36) {
            historyItem.typeSuccessStatus[type.id] = false;
            return;
        }

        const terminals = terminalMapping?.[baseNum] || [];
        const hitZone = getHitZone(baseNum, terminals, winningNumber, useDynamicTerminalNeighbourCountBool, terminalMapping, rouletteWheel);

        if (hitZone.includes(winningNumber)) {
            historyItem.hitTypes.push(type.id);
            historyItem.typeSuccessStatus[type.id] = true;
            let currentMinDist = Infinity;
            hitZone.forEach(zoneNum => {
                const dist = calculatePocketDistance(zoneNum, winningNumber, rouletteWheel);
                if (dist < currentMinDist) currentMinDist = dist;
            });
            if (currentMinDist < minPocketDistance) minPocketDistance = currentMinDist;
        } else {
            historyItem.typeSuccessStatus[type.id] = false;
        }
    });

    historyItem.status = historyItem.hitTypes.length > 0 ? 'success' : 'fail';
    historyItem.pocketDistance = minPocketDistance !== Infinity ? minPocketDistance : null;

    if (historyItem.recommendedGroupId && historyItem.hitTypes.includes(historyItem.recommendedGroupId)) {
        historyItem.recommendedGroupPocketDistance = historyItem.pocketDistance;
    } else {
        historyItem.recommendedGroupPocketDistance = null;
    }
}

function calculateTrendStats(currentHistory, current_STRATEGY_CONFIG, activeTypesArr) {
    const sortedHistory = [...currentHistory].sort((a, b) => a.id - b.id);
    const streakData = {};
    const currentStreaks = {};
    const totalOccurrences = {};
    const successfulOccurrences = {};
    let lastSuccessState = [];

    activeTypesArr.forEach(type => {
        streakData[type.id] = [];
        currentStreaks[type.id] = 0;
        totalOccurrences[type.id] = 0;
        successfulOccurrences[type.id] = 0;
    });

    sortedHistory.forEach((item, i) => {
        if (item.status === 'pending') return;
        const weight = Math.pow(current_STRATEGY_CONFIG.decayFactor, sortedHistory.length - 1 - i);
        activeTypesArr.forEach(type => {
            if (item.typeSuccessStatus && item.typeSuccessStatus.hasOwnProperty(type.id)) {
                 totalOccurrences[type.id] += weight;
            }
            if (item.typeSuccessStatus && item.typeSuccessStatus[type.id]) {
                currentStreaks[type.id]++;
                successfulOccurrences[type.id] += weight;
            } else {
                if (currentStreaks[type.id] > 0) streakData[type.id].push(currentStreaks[type.id]);
                currentStreaks[type.id] = 0;
            }
        });
        if (item.status === 'success') lastSuccessState = item.hitTypes;
    });

    const averages = {};
    activeTypesArr.forEach(type => {
        const allStreaks = [...streakData[type.id]];
        if (currentStreaks[type.id] > 0) allStreaks.push(currentStreaks[type.id]);
        averages[type.id] = allStreaks.length > 0 ? (allStreaks.reduce((a, b) => a + b, 0) / allStreaks.length) : 0;
    });

    return { averages, currentStreaks, lastSuccessState };
}

function getBoardStateStats(simulatedHistory, current_STRATEGY_CONFIG, activePredictionTypes) {
    const stats = {};
    activePredictionTypes.forEach(type => {
        stats[type.id] = { success: 0, total: 0 };
    });
    simulatedHistory.forEach((item, i) => {
        const weight = Math.pow(current_STRATEGY_CONFIG.decayFactor, simulatedHistory.length - 1 - i);
        activePredictionTypes.forEach(type => {
            if (item.typeSuccessStatus && item.typeSuccessStatus.hasOwnProperty(type.id)) {
                stats[type.id].total += weight;
            }
        });
        if (item.status === 'success') {
            item.hitTypes.forEach(typeId => {
                if (stats[typeId]) stats[typeId].success += weight;
            });
        }
    });
    return stats;
}

function runNeighbourAnalysis(simulatedHistory, current_STRATEGY_CONFIG, useDynamicTerminalNeighbourCountBool, allPredictionTypes, terminalMapping, rouletteWheel) {
    const analysis = {};
    for (let i = 0; i <= 36; i++) analysis[i] = { success: 0 };
    simulatedHistory.forEach((item, i) => {
        if (item.status !== 'success') return;
        const weight = Math.pow(current_STRATEGY_CONFIG.decayFactor, simulatedHistory.length - 1 - i);
        item.hitTypes.forEach(typeId => {
            const type = allPredictionTypes.find(t => t.id === typeId);
            if (!type) return;
            const baseNum = type.calculateBase(item.num1, item.num2);
            if (baseNum < 0 || baseNum > 36) return;
            const hitZone = getHitZone(baseNum, terminalMapping[baseNum] || [], item.winningNumber, useDynamicTerminalNeighbourCountBool, terminalMapping, rouletteWheel);
            hitZone.forEach(num => {
                if (analysis[num]) analysis[num].success += weight;
            });
        });
    });
    return analysis;
}

function getRecommendation(context) {
    const {
        trendStats, boardStats, neighbourScores,
        diff, inputNum1, inputNum2,
        isForWeightUpdate = false,
        currentAdaptiveInfluences, lastWinningNumber,
        useProximityBoostBool, useWeightedZoneBool, useNeighbourFocusBool,
        isAiReadyBool, useTrendConfirmationBool,
        current_STRATEGY_CONFIG,
        activePredictionTypes, allPredictionTypes, terminalMapping, rouletteWheel
    } = context;

    let candidates = activePredictionTypes.map(type => {
        const details = {
            hitRate: (boardStats[type.id]?.total > 0 ? (boardStats[type.id]?.success / boardStats[type.id]?.total * 100) : 0),
            avgTrend: trendStats.averages[type.id] || 0,
            currentStreak: trendStats.currentStreaks[type.id] || 0,
            predictiveDistance: Infinity,
            proximityBoostApplied: false,
            weightedZoneBoostApplied: false,
            finalScore: 0,
            primaryDrivingFactor: "N/A",
            individualScores: {}
        };
        const baseNum = type.calculateBase(inputNum1, inputNum2);
        if (baseNum < 0 || baseNum > 36) return null;
        const terminals = terminalMapping?.[baseNum] || [];
        const hitZone = getHitZone(baseNum, terminals, lastWinningNumber, context.useDynamicTerminalNeighbourCount, terminalMapping, rouletteWheel);
        let rawScore = 0;
        const rawHitRatePoints = Math.max(0, details.hitRate - 40) * 0.5;
        rawScore += rawHitRatePoints;
        details.individualScores['Hit Rate'] = rawHitRatePoints;
        const rawStreakPoints = Math.min(15, details.currentStreak * 5);
        rawScore += rawStreakPoints;
        details.individualScores['Streak'] = rawStreakPoints;
        if (useProximityBoostBool && lastWinningNumber !== null) {
            hitZone.forEach(zoneNum => {
                const dist = calculatePocketDistance(zoneNum, lastWinningNumber, rouletteWheel);
                if (dist < details.predictiveDistance) details.predictiveDistance = dist;
            });
            if (details.predictiveDistance <= 5) {
                const rawProximityPoints = (5 - details.predictiveDistance) * 2;
                rawScore += rawProximityPoints;
                details.individualScores['Proximity to Last Spin'] = rawProximityPoints;
            }
        }
        if (useWeightedZoneBool) {
            const neighbourWeightedScore = hitZone.reduce((sum, num) => sum + (neighbourScores[num]?.success || 0), 0);
            const rawNeighbourPoints = Math.min(10, neighbourWeightedScore * 0.5);
            rawScore += rawNeighbourPoints;
            details.individualScores['Hot Zone Weighting'] = rawNeighbourPoints;
        }
        let finalCalculatedScore = 0;
        let mostInfluentialFactor = "N/A";
        let highestInfluencedScore = 0;
        for (const factorName in currentAdaptiveInfluences) {
            const influence = currentAdaptiveInfluences[factorName];
            let factorScore = details.individualScores[factorName] || 0;
            const influencedScore = factorScore * influence;
            finalCalculatedScore += influencedScore;
            if (influencedScore > highestInfluencedScore) {
                highestInfluencedScore = influencedScore;
                mostInfluentialFactor = factorName;
            }
        }
        details.finalScore = finalCalculatedScore;
        details.primaryDrivingFactor = mostInfluentialFactor;
        return { type, score: details.finalScore, details };
    }).filter(c => c && !isNaN(c.score));
    if (candidates.length === 0) return { html: '<span>Wait for Signal</span>', bestCandidate: null, details: null };
    candidates.sort((a, b) => b.score - a.score);
    let bestCandidate = candidates[0];
    if (bestCandidate.score <= 0) return { html: '<span>Wait for Signal</span>', bestCandidate: null, details: null };
    if (isForWeightUpdate) return { bestCandidate };
    let signal = "Play";
    let signalColor = "text-purple-700";
    if (bestCandidate.score > 50) {
        signal = "Strong Play";
        signalColor = "text-green-600";
    }
    if (useTrendConfirmationBool && trendStats.lastSuccessState.length > 0 && !trendStats.lastSuccessState.includes(bestCandidate.type.id)) {
        signal = 'Wait';
        signalColor = "text-gray-500";
    }
    const reason = `(${bestCandidate.details?.primaryDrivingFactor || 'Statistical Trends'})`;
    let finalHtml = `<strong class="${signalColor}">${signal}:</strong> Play <strong style="color: ${bestCandidate.type.textColor};">${bestCandidate.type.label}</strong><br><span class="text-xs">${reason}</span>`;
    if (signal === 'Wait') {
        finalHtml = `<strong class="${signalColor}">${signal}</strong>`;
    }
    return { html: finalHtml, bestCandidate, details: bestCandidate.details };
}