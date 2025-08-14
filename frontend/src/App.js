import React, { useState, useEffect } from 'react';
import axios from 'axios';
import IpcChart from './components/IpcChart';

const API_BASE = 'http://localhost:8000';

function App() {
  const [countries, setCountries] = useState([]);
  const [levels, setLevels] = useState([]);
  const [areas, setAreas] = useState([]);
  const [selectedCountry, setSelectedCountry] = useState('');
  const [selectedLevel, setSelectedLevel] = useState('');
  const [selectedArea, setSelectedArea] = useState('');
  const [graphData, setGraphData] = useState([]);

  useEffect(() => {
    axios.get(`${API_BASE}/countries`).then(res => setCountries(res.data.countries));
  }, []);

  useEffect(() => {
    if (selectedCountry) {
      axios.get(`${API_BASE}/levels/${selectedCountry}`).then(res => setLevels(res.data.levels));
      setSelectedLevel('');
      setSelectedArea('');
      setAreas([]);
    }
  }, [selectedCountry]);

  useEffect(() => {
    if (selectedCountry && selectedLevel) {
      axios.get(`${API_BASE}/areas/${selectedCountry}/${selectedLevel}`).then(res => setAreas(res.data.areas));
      setSelectedArea('');
    }
  }, [selectedLevel]);

  const loadGraph = () => {
    let params = `country=${selectedCountry}`;
    if (selectedLevel) params += `&level1=${selectedLevel}`;
    if (selectedArea) params += `&area=${selectedArea}`;
    axios.get(`${API_BASE}/graph-data?${params}`).then(res => setGraphData(res.data));
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">IPC Data Visualization</h1>
      <div className="flex space-x-4 mb-4">
        <select
          className="border p-2 rounded"
          value={selectedCountry}
          onChange={e => setSelectedCountry(e.target.value)}
        >
          {/* Graph Dropdown for Country */}
          <option value="">Select Country</option>
          {countries.map(c => <option key={c} value={c}>{c}</option>)}
        </select>
        {levels.length > 0 && (
          <select
            className="border p-2 rounded"
            value={selectedLevel}
            onChange={e => setSelectedLevel(e.target.value)}
          >
            {/* Graph Dropdown for Region */}
            <option value="">Select Region</option>
            {levels.map(l => <option key={l} value={l}>{l}</option>)}
          </select>
        )}
        {areas.length > 0 && (
          <select
            className="border p-2 rounded"
            value={selectedArea}
            onChange={e => setSelectedArea(e.target.value)}
          >
            {/* Graph Dropdown for Area */}
            <option value="">Select Area</option>
            {areas.map(a => <option key={a} value={a}>{a}</option>)}
          </select>
        )}
        {/* Load Graph Button */}
        <button
          className="bg-blue-500 text-white p-2 rounded"
          onClick={loadGraph}
          disabled={!selectedCountry}
        >
          Load Graph
        </button>
      </div>
      {graphData.length > 0 && <IpcChart data={graphData} />}
    </div>
  );
}

export default App;
