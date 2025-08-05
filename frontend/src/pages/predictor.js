import React, { useState, useEffect } from "react";

const API_BASE = "http://localhost:5001";
export default function Predictor() {
  const [upcomingPredictions, setUpcomingPredictions] = useState([]);
  const [historicalPredictions, setHistoricalPredictions] = useState([]);
  const [availableRaces, setAvailableRaces] = useState({});
  const [availableSeasons, setAvailableSeasons] = useState([]);
  const [selectedYear, setSelectedYear] = useState("");
  const [selectedRace, setSelectedRace] = useState("");
  const [loading, setLoading] = useState(false);
  const [upcomingLoading, setUpcomingLoading] = useState(true);
  const [racesLoading, setRacesLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    loadAvailableRaces();
    loadUpcomingPredictions();
  }, []);

  const loadAvailableRaces = async () => {
    try {
      setRacesLoading(true);

      // Try to use your existing backend structure
      const response = await fetch(`${API_BASE}/available_races`);

      if (response.ok) {
        const data = await response.json();
        console.log("Loaded races:", data);

        setAvailableRaces(data.races);
        setAvailableSeasons(data.seasons);

        // Set default year to the most recent season
        if (data.seasons.length > 0) {
          setSelectedYear(data.seasons[data.seasons.length - 1].toString());
        }
      } else {
        throw new Error("Backend endpoint not available");
      }
    } catch (err) {
      console.error("Error loading available races:", err);

      // Fallback: Create basic structure based on common F1 calendar
      const fallbackRaces = {
        2024: [
          { id: "bahrain", round: 1, name: "Bahrain Grand Prix" },
          { id: "jeddah", round: 2, name: "Saudi Arabian Grand Prix" },
          { id: "albert_park", round: 3, name: "Australian Grand Prix" },
          { id: "suzuka", round: 4, name: "Japanese Grand Prix" },
          { id: "shanghai", round: 5, name: "Chinese Grand Prix" },
          { id: "miami", round: 6, name: "Miami Grand Prix" },
          { id: "imola", round: 7, name: "Emilia Romagna Grand Prix" },
          { id: "monaco", round: 8, name: "Monaco Grand Prix" },
          { id: "villeneuve", round: 9, name: "Canadian Grand Prix" },
          { id: "catalunya", round: 10, name: "Spanish Grand Prix" },
          { id: "red_bull_ring", round: 11, name: "Austrian Grand Prix" },
          { id: "silverstone", round: 12, name: "British Grand Prix" },
          { id: "hungaroring", round: 13, name: "Hungarian Grand Prix" },
          { id: "spa", round: 14, name: "Belgian Grand Prix" },
          { id: "zandvoort", round: 15, name: "Dutch Grand Prix" },
          { id: "monza", round: 16, name: "Italian Grand Prix" },
          { id: "baku", round: 17, name: "Azerbaijan Grand Prix" },
          { id: "marina_bay", round: 18, name: "Singapore Grand Prix" },
          { id: "americas", round: 19, name: "United States Grand Prix" },
          { id: "rodriguez", round: 20, name: "Mexican Grand Prix" },
          { id: "interlagos", round: 21, name: "Brazilian Grand Prix" },
          { id: "las_vegas", round: 22, name: "Las Vegas Grand Prix" },
          { id: "lusail", round: 23, name: "Qatar Grand Prix" },
          { id: "yas_marina", round: 24, name: "Abu Dhabi Grand Prix" },
        ],
        2025: [],
      };

      setAvailableRaces(fallbackRaces);
      setAvailableSeasons([2024, 2025]);
      setSelectedYear("2024");
    } finally {
      setRacesLoading(false);
    }
  };

  // Reset selected race when year changes
  useEffect(() => {
    setSelectedRace("");
    setHistoricalPredictions([]);
  }, [selectedYear]);

  const loadUpcomingPredictions = async () => {
    try {
      setUpcomingLoading(true);

      // Use the existing endpoint with next_only=true parameter
      const response = await fetch(
        `${API_BASE}/predict_race_with_predicted_grid?season=2025&confidence=false&next_only=true`
      );

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();

      if (data.predictions && data.predictions.length > 0) {
        setUpcomingPredictions(data.predictions);
      } else {
        setUpcomingPredictions([]);
      }
    } catch (err) {
      console.error("Error loading upcoming predictions:", err);
      setUpcomingPredictions([]);
    } finally {
      setUpcomingLoading(false);
    }
  };

  // Load historical predictions when race is selected
  const loadHistoricalPredictions = React.useCallback(async () => {
    if (!selectedRace) return;

    try {
      setLoading(true);
      setError("");

      const currentRaces = availableRaces[selectedYear] || [];
      const raceData = currentRaces.find((race) => race.id === selectedRace);

      if (!raceData) {
        throw new Error("Race not found");
      }

      // Load THREE types of predictions for completed races:
      // 1. Post-qualifying predictions (using actual grid)
      // 2. Historical two-stage predictions (predicted grid → race, ignoring actual qualifying)
      // 3. Grid position predictions vs actual

      // 1. Post-qualifying predictions (original)
      const finishResponse = await fetch(
        `${API_BASE}/predict_finish_pos_round?season=${selectedYear}&round=${raceData.round}&confidence=false`
      );

      let postQualPredictions = [];
      if (finishResponse.ok) {
        const finishData = await finishResponse.json();
        postQualPredictions = finishData.predictions || [];
      }

      // 2. Historical two-stage predictions (NEW - ignores actual qualifying)
      const historicalTwoStageResponse = await fetch(
        `${API_BASE}/predict_historical_race_two_stage?season=${selectedYear}&round=${raceData.round}&confidence=false`
      );

      let historicalTwoStagePredictions = [];
      if (historicalTwoStageResponse.ok) {
        const historicalTwoStageData = await historicalTwoStageResponse.json();
        historicalTwoStagePredictions =
          historicalTwoStageData.predictions || [];
      }

      // 3. Grid comparison
      const gridComparisonResponse = await fetch(
        `${API_BASE}/compare_grid_predictions?season=${selectedYear}&round=${raceData.round}`
      );

      let gridComparison = [];
      if (gridComparisonResponse.ok) {
        const gridData = await gridComparisonResponse.json();
        gridComparison = gridData.grid_comparison || [];
      }

      // Combine all data for display
      const combinedData = postQualPredictions.map((postQual) => {
        const historicalTwoStage = historicalTwoStagePredictions.find(
          (hts) => hts.driver === postQual.driver
        );
        const gridComp = gridComparison.find(
          (gc) => gc.driver === postQual.driver
        );

        return {
          ...postQual,
          historical_two_stage_predicted:
            historicalTwoStage?.predicted_finish || null,
          historical_predicted_grid: historicalTwoStage?.predicted_grid || null,
          grid_comparison_predicted: gridComp?.predicted_grid || null,
          actual_grid:
            gridComp?.actual_grid || historicalTwoStage?.actual_grid || null,
        };
      });

      setHistoricalPredictions(combinedData);
    } catch (err) {
      console.error("Error loading historical predictions:", err);
      setError(`Failed to load predictions: ${err.message}`);
      setHistoricalPredictions([]);
    } finally {
      setLoading(false);
    }
  }, [selectedRace, selectedYear, availableRaces]);

  // Load historical predictions when race is selected
  useEffect(() => {
    loadHistoricalPredictions();
  }, [loadHistoricalPredictions]);

  const getPositionColor = (position) => {
    if (position === 1) return "text-yellow-400"; // Gold
    if (position === 2) return "text-gray-300"; // Silver
    if (position === 3) return "text-orange-400"; // Bronze
    if (position <= 10) return "text-green-400"; // Points
    return "text-f1-white";
  };

  const getUpcomingRaceName = () => {
    if (upcomingPredictions.length > 0) {
      const round = upcomingPredictions[0].round;
      const currentRaces = availableRaces[2025] || [];
      const race = currentRaces.find((r) => r.round === round);
      return race ? `${race.name} (Round ${round})` : `Round ${round}`;
    }
    return "Next Race";
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h2 className="text-4xl font-f1-bold text-f1-red mb-8">
        Race Predictions
      </h2>

      {/* Upcoming Race Predictions */}
      <div className="mb-12">
        <h3 className="text-2xl font-f1-primary mb-4 text-f1-white">
          🏁 {getUpcomingRaceName()} - 2025
        </h3>

        {upcomingLoading ? (
          <div className="bg-gray-800 rounded-lg p-6 text-center">
            <div className="animate-pulse text-f1-white">
              Loading upcoming race predictions...
            </div>
          </div>
        ) : upcomingPredictions.length > 0 ? (
          <div className="bg-gray-800 rounded-lg overflow-hidden shadow-lg">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-f1-red">
                  <tr>
                    <th className="px-4 py-3 text-left font-f1-bold">Driver</th>
                    <th className="px-4 py-3 text-left font-f1-bold">
                      Pred. Grid
                    </th>
                    <th className="px-4 py-3 text-left font-f1-bold">
                      Pred. Finish
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {upcomingPredictions.map((pred, idx) => (
                    <tr
                      key={idx}
                      className="border-b border-gray-700 hover:bg-gray-700"
                    >
                      <td className="px-4 py-3 font-f1-primary text-f1-white">
                        {pred.driver}
                      </td>
                      <td
                        className={`px-4 py-3 font-f1-bold ${getPositionColor(
                          pred.predicted_grid
                        )}`}
                      >
                        P{Math.round(pred.predicted_grid)}
                      </td>
                      <td
                        className={`px-4 py-3 font-f1-bold text-lg ${getPositionColor(
                          pred.predicted_finish
                        )}`}
                      >
                        P{Math.round(pred.predicted_finish)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div className="bg-gray-800 rounded-lg p-6 text-center text-gray-400">
            No upcoming race predictions available
          </div>
        )}
      </div>

      {/* Historical Predictions Section */}
      <div>
        <h3 className="text-2xl font-f1-primary mb-6 text-f1-white">
          📊 Historical Race Analysis
        </h3>

        {/* Controls */}
        <div className="flex flex-wrap gap-4 mb-6">
          {/* Year Selector */}
          <div>
            <label className="block text-sm font-f1-primary text-gray-300 mb-2">
              Season
            </label>
            <select
              value={selectedYear}
              onChange={(e) => setSelectedYear(e.target.value)}
              className="bg-gray-800 text-f1-white border border-gray-600 rounded px-3 py-2 font-f1-primary focus:border-f1-red focus:outline-none"
              disabled={racesLoading}
            >
              {racesLoading ? (
                <option>Loading seasons...</option>
              ) : (
                availableSeasons.map((season) => (
                  <option key={season} value={season.toString()}>
                    {season}
                  </option>
                ))
              )}
            </select>
          </div>

          {/* Race Selector */}
          <div>
            <label className="block text-sm font-f1-primary text-gray-300 mb-2">
              Grand Prix
            </label>
            <select
              value={selectedRace}
              onChange={(e) => setSelectedRace(e.target.value)}
              className="bg-gray-800 text-f1-white border border-gray-600 rounded px-3 py-2 font-f1-primary focus:border-f1-red focus:outline-none min-w-64"
              disabled={racesLoading || !selectedYear}
            >
              <option value="">
                {racesLoading ? "Loading races..." : "Select a race..."}
              </option>
              {!racesLoading &&
                availableRaces[selectedYear]?.map((race) => (
                  <option key={race.id} value={race.id}>
                    Round {race.round}: {race.name}
                  </option>
                ))}
            </select>
          </div>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="bg-gray-800 rounded-lg p-6 text-center">
            <div className="animate-pulse text-f1-white">
              Loading predictions...
            </div>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="bg-red-900 border border-red-600 rounded-lg p-4 mb-6">
            <p className="text-red-200">{error}</p>
          </div>
        )}

        {/* Historical Predictions Table */}
        {!loading && selectedRace && (
          <div className="bg-gray-800 rounded-lg overflow-hidden shadow-lg">
            {historicalPredictions.length > 0 && (
              <div className="overflow-x-auto">
                <div className="bg-f1-red px-4 py-2">
                  <h4 className="font-f1-bold text-f1-white">
                    Race Prediction Comparison
                  </h4>
                  <p className="text-sm text-f1-white opacity-90">
                    Post-Qualifying vs Two-Stage (Grid→Race) Predictions
                  </p>
                </div>
                <table className="w-full">
                  <thead className="bg-gray-700">
                    <tr>
                      <th className="px-3 py-3 text-left font-f1-primary text-sm">
                        Driver
                      </th>
                      <th className="px-3 py-3 text-left font-f1-primary text-sm">
                        Pred Grid
                      </th>
                      <th className="px-3 py-3 text-left font-f1-primary text-sm">
                        Actual Grid
                      </th>
                      <th className="px-3 py-3 text-left font-f1-primary text-sm">
                        Post-Qual
                      </th>
                      <th className="px-3 py-3 text-left font-f1-primary text-sm">
                        Two-Stage
                      </th>
                      <th className="px-3 py-3 text-left font-f1-primary text-sm">
                        Actual Finish
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {historicalPredictions.map((pred, idx) => (
                      <tr
                        key={idx}
                        className="border-b border-gray-700 hover:bg-gray-700"
                      >
                        <td className="px-3 py-3 font-f1-primary text-f1-white text-sm">
                          {pred.driver}
                        </td>
                        <td
                          className={`px-3 py-3 font-f1-bold text-sm ${
                            pred.historical_predicted_grid
                              ? getPositionColor(pred.historical_predicted_grid)
                              : "text-gray-500"
                          }`}
                        >
                          {pred.historical_predicted_grid
                            ? `P${Math.round(pred.historical_predicted_grid)}`
                            : "N/A"}
                        </td>
                        <td
                          className={`px-3 py-3 font-f1-bold text-sm ${
                            pred.actual_grid
                              ? getPositionColor(pred.actual_grid)
                              : "text-gray-500"
                          }`}
                        >
                          {pred.actual_grid
                            ? `P${Math.round(pred.actual_grid)}`
                            : "N/A"}
                        </td>
                        <td
                          className={`px-3 py-3 font-f1-bold text-sm ${getPositionColor(
                            pred.predicted
                          )}`}
                        >
                          P{Math.round(pred.predicted)}
                        </td>
                        <td
                          className={`px-3 py-3 font-f1-bold text-sm ${
                            pred.historical_two_stage_predicted
                              ? getPositionColor(
                                  pred.historical_two_stage_predicted
                                )
                              : "text-gray-500"
                          }`}
                        >
                          {pred.historical_two_stage_predicted
                            ? `P${Math.round(
                                pred.historical_two_stage_predicted
                              )}`
                            : "N/A"}
                        </td>
                        <td
                          className={`px-3 py-3 font-f1-bold text-sm ${
                            pred.actual
                              ? getPositionColor(pred.actual)
                              : "text-gray-500"
                          }`}
                        >
                          {pred.actual ? `P${Math.round(pred.actual)}` : "N/A"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="bg-gray-700 px-4 py-2 text-xs text-gray-300">
                  <p>
                    <strong>Post-Qual:</strong> Uses actual qualifying results |{" "}
                    <strong>Two-Stage:</strong> Predicts grid first, then race
                    result
                  </p>
                </div>
              </div>
            )}

            {!loading && selectedRace && historicalPredictions.length === 0 && (
              <div className="p-6 text-center text-gray-400">
                No predictions available for this race
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
