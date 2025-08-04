import React, { useState, useEffect } from 'react';

const API_BASE = 'http://localhost:5001'; // Adjust if needed

export default function Predictor() {
  const [upcomingPredictions, setUpcomingPredictions] = useState([]);
  const [historicalPredictions, setHistoricalPredictions] = useState([]);
  
  const [availableRaces, setAvailableRaces] = useState({});
  const [availableSeasons, setAvailableSeasons] = useState([]);
  const [selectedYear, setSelectedYear] = useState('');
  const [selectedRace, setSelectedRace] = useState('');
  const [loading, setLoading] = useState(false);
  const [upcomingLoading, setUpcomingLoading] = useState(true);
  const [racesLoading, setRacesLoading] = useState(true);
  const [error, setError] = useState('');
  const [viewMode] = useState('finish'); // 'finish' or 'fastest'

  // Load available races from CSV data on component mount
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
        console.log('Loaded races:', data);
        
        setAvailableRaces(data.races);
        setAvailableSeasons(data.seasons);
        
        // Set default year to the most recent season
        if (data.seasons.length > 0) {
          setSelectedYear(data.seasons[data.seasons.length - 1].toString());
        }
        
      } else {
        throw new Error('Backend endpoint not available');
      }
      
    } catch (err) {
      console.error('Error loading available races:', err);
      
      // Fallback: Create basic structure based on common F1 calendar
      const fallbackRaces = {
        '2024': [
          { id: 'bahrain', round: 1, name: 'Bahrain Grand Prix' },
          { id: 'jeddah', round: 2, name: 'Saudi Arabian Grand Prix' },
          { id: 'albert_park', round: 3, name: 'Australian Grand Prix' },
          { id: 'suzuka', round: 4, name: 'Japanese Grand Prix' },
          { id: 'shanghai', round: 5, name: 'Chinese Grand Prix' },
          { id: 'miami', round: 6, name: 'Miami Grand Prix' },
          { id: 'imola', round: 7, name: 'Emilia Romagna Grand Prix' },
          { id: 'monaco', round: 8, name: 'Monaco Grand Prix' },
          { id: 'villeneuve', round: 9, name: 'Canadian Grand Prix' },
          { id: 'catalunya', round: 10, name: 'Spanish Grand Prix' },
          { id: 'red_bull_ring', round: 11, name: 'Austrian Grand Prix' },
          { id: 'silverstone', round: 12, name: 'British Grand Prix' },
          { id: 'hungaroring', round: 13, name: 'Hungarian Grand Prix' },
          { id: 'spa', round: 14, name: 'Belgian Grand Prix' },
          { id: 'zandvoort', round: 15, name: 'Dutch Grand Prix' },
          { id: 'monza', round: 16, name: 'Italian Grand Prix' },
          { id: 'baku', round: 17, name: 'Azerbaijan Grand Prix' },
          { id: 'marina_bay', round: 18, name: 'Singapore Grand Prix' },
          { id: 'americas', round: 19, name: 'United States Grand Prix' },
          { id: 'rodriguez', round: 20, name: 'Mexican Grand Prix' },
          { id: 'interlagos', round: 21, name: 'Brazilian Grand Prix' },
          { id: 'las_vegas', round: 22, name: 'Las Vegas Grand Prix' },
          { id: 'lusail', round: 23, name: 'Qatar Grand Prix' },
          { id: 'yas_marina', round: 24, name: 'Abu Dhabi Grand Prix' }
        ],
        '2025': []
      };
      
      setAvailableRaces(fallbackRaces);
      setAvailableSeasons([2024, 2025]);
      setSelectedYear('2024');
      
    } finally {
      setRacesLoading(false);
    }
  };

  // Reset selected race when year changes
  useEffect(() => {
    setSelectedRace('');
      setHistoricalPredictions([]);
      
  }, [selectedYear]);

  const loadUpcomingPredictions = async () => {
    try {
      setUpcomingLoading(true);
      const response = await fetch(`${API_BASE}/predict_finish_pos_future?season=2025&confidence=true`);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.predictions && data.predictions.length > 0) {
        // Group by round and get the next upcoming race
        const rounds = [...new Set(data.predictions.map(p => p.round))].sort((a, b) => a - b);
        const nextRound = rounds[0]; // Get the first/next round
        const nextRacePredictions = data.predictions
          .filter(p => p.round === nextRound)
          .sort((a, b) => a.predicted - b.predicted);
        
        setUpcomingPredictions(nextRacePredictions);
      } else {
        setUpcomingPredictions([]);
      }
    } catch (err) {
      console.error('Error loading upcoming predictions:', err);
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
      setError('');
      
      const currentRaces = availableRaces[selectedYear] || [];
      const raceData = currentRaces.find(race => race.id === selectedRace);
      
      if (!raceData) {
        throw new Error('Race not found');
      }

      // Load finish position predictions
      const finishResponse = await fetch(
        `${API_BASE}/predict_finish_pos_round?season=${selectedYear}&round=${raceData.round}&confidence=true`
      );
      
      if (finishResponse.ok) {
        const finishData = await finishResponse.json();
        setHistoricalPredictions(finishData.predictions || []);
      } else {
        setHistoricalPredictions([]);
      }

     
      
    } catch (err) {
      console.error('Error loading historical predictions:', err);
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
    if (position === 1) return 'text-yellow-400'; // Gold
    if (position === 2) return 'text-gray-300';   // Silver
    if (position === 3) return 'text-orange-400'; // Bronze
    if (position <= 10) return 'text-green-400';  // Points
    return 'text-f1-white';
  };

  
  const getUpcomingRaceName = () => {
    if (upcomingPredictions.length > 0) {
      const round = upcomingPredictions[0].round;
      const currentRaces = availableRaces[2025] || [];
      const race = currentRaces.find(r => r.round === round);
      return race ? `${race.name} (Round ${round})` : `Round ${round}`;
    }
    return 'Next Race';
  };

  
  return (
    <div className="max-w-6xl mx-auto p-6">
      <h2 className="text-4xl font-f1-bold text-f1-red mb-8">Race Predictions</h2>
      
      {/* Upcoming Race Predictions */}
      <div className="mb-12">
        <h3 className="text-2xl font-f1-primary mb-4 text-f1-white">
          🏁 {getUpcomingRaceName()} - 2025
        </h3>
        
        {upcomingLoading ? (
          <div className="bg-gray-800 rounded-lg p-6 text-center">
            <div className="animate-pulse text-f1-white">Loading upcoming race predictions...</div>
          </div>
        ) : upcomingPredictions.length > 0 ? (
          <div className="bg-gray-800 rounded-lg overflow-hidden shadow-lg">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-f1-red">
                  <tr>
                    <th className="px-4 py-3 text-left font-f1-bold">Position</th>
                    <th className="px-4 py-3 text-left font-f1-bold">Driver</th>
                    <th className="px-4 py-3 text-left font-f1-bold">Score</th>
                    <th className="px-4 py-3 text-left font-f1-bold">Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {upcomingPredictions.map((pred, idx) => (
                    <tr key={idx} className="border-b border-gray-700 hover:bg-gray-700">
                      <td className={`px-4 py-3 font-f1-bold text-lg ${getPositionColor(pred.predicted)}`}>
                        P{Math.round(pred.predicted)}
                      </td>
                      <td className="px-4 py-3 font-f1-primary text-f1-white">{pred.driver}</td>
                      <td className="px-4 py-3 text-gray-300">{pred.model_score?.toFixed(3)}</td>
                      <td className="px-4 py-3 text-gray-300">
                        {pred.confidence ? `±${pred.confidence.toFixed(2)}` : 'N/A'}
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
        <h3 className="text-2xl font-f1-primary mb-6 text-f1-white">📊 Historical Race Analysis</h3>
        
        {/* Controls */}
        <div className="flex flex-wrap gap-4 mb-6">
          {/* Year Selector */}
          <div>
            <label className="block text-sm font-f1-primary text-gray-300 mb-2">Season</label>
            <select 
              value={selectedYear} 
              onChange={(e) => setSelectedYear(e.target.value)}
              className="bg-gray-800 text-f1-white border border-gray-600 rounded px-3 py-2 font-f1-primary focus:border-f1-red focus:outline-none"
              disabled={racesLoading}
            >
              {racesLoading ? (
                <option>Loading seasons...</option>
              ) : (
                availableSeasons.map(season => (
                  <option key={season} value={season.toString()}>{season}</option>
                ))
              )}
            </select>
          </div>

          {/* Race Selector */}
          <div>
            <label className="block text-sm font-f1-primary text-gray-300 mb-2">Grand Prix</label>
            <select 
              value={selectedRace} 
              onChange={(e) => setSelectedRace(e.target.value)}
              className="bg-gray-800 text-f1-white border border-gray-600 rounded px-3 py-2 font-f1-primary focus:border-f1-red focus:outline-none min-w-64"
              disabled={racesLoading || !selectedYear}
            >
              <option value="">
                {racesLoading ? "Loading races..." : "Select a race..."}
              </option>
              {!racesLoading && availableRaces[selectedYear]?.map((race) => (
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
            <div className="animate-pulse text-f1-white">Loading predictions...</div>
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
            {viewMode === 'finish' && historicalPredictions.length > 0 && (
              <div className="overflow-x-auto">
                <div className="bg-f1-red px-4 py-2">
                  <h4 className="font-f1-bold text-f1-white">Finish Position Predictions</h4>
                </div>
                <table className="w-full">
                  <thead className="bg-gray-700">
                    <tr>
                      <th className="px-4 py-3 text-left font-f1-primary">Predicted</th>
                      <th className="px-4 py-3 text-left font-f1-primary">Driver</th>
                      <th className="px-4 py-3 text-left font-f1-primary">Actual</th>
                      <th className="px-4 py-3 text-left font-f1-primary">Score</th>
                      <th className="px-4 py-3 text-left font-f1-primary">Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {historicalPredictions.map((pred, idx) => (
                      <tr key={idx} className="border-b border-gray-700 hover:bg-gray-700">
                        <td className={`px-4 py-3 font-f1-bold ${getPositionColor(pred.predicted)}`}>
                          P{Math.round(pred.predicted)}
                        </td>
                        <td className="px-4 py-3 font-f1-primary text-f1-white">{pred.driver}</td>
                        <td className={`px-4 py-3 font-f1-bold ${pred.actual ? getPositionColor(pred.actual) : 'text-gray-500'}`}>
                          {pred.actual ? `P${Math.round(pred.actual)}` : 'N/A'}
                        </td>
                        <td className="px-4 py-3 text-gray-300">{pred.model_score?.toFixed(3)}</td>
                        <td className="px-4 py-3 text-gray-300">
                          {pred.confidence ? `±${pred.confidence.toFixed(2)}` : 'N/A'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}