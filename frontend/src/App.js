// Node modules are located in the frontend directory
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import IpcChart from './components/IpcChart';

import { motion } from 'framer-motion';
import Select from 'react-select';

const API_BASE = 'http://localhost:8000';

function App() {
  const [countries, setCountries] = useState([]);
  const [levels, setLevels] = useState([]);
  const [areas, setAreas] = useState([]);
  const [selectedCountry, setSelectedCountry] = useState('');
  const [selectedLevel, setSelectedLevel] = useState('');
  const [selectedArea, setSelectedArea] = useState('');
  const [graphData, setGraphData] = useState([]);

  const [activeTab, setActiveTab] = useState('pulse');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [countriesLoading, setCountriesLoading] = useState(true);

  useEffect(() => {
    const fetchCountries = async () => {
      try {
        const res = await axios.get(`${API_BASE}/countries`);
        setCountries(res.data.countries);
      } catch (err) {
        setError('Failed to load countries. Is the backend running?');
      } finally {
        setCountriesLoading(false);
      }
    };
    fetchCountries();
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

  const loadGraph = async () => {
    setLoading(true);
    setError('');
    try {
      let params = `country=${encodeURIComponent(selectedCountry)}`;
      if (selectedLevel) params += `&level1=${encodeURIComponent(selectedLevel)}`;
      if (selectedArea) params += `&area=${encodeURIComponent(selectedArea)}`;
      const res = await axios.get(`${API_BASE}/graph-data?${params}`);
      setGraphData(res.data);
    } catch (err) {
      setError('Failed to load data. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50">
      <header className="bg-white shadow-md">
        <div className="max-w-7xl mx-auto px-4 py-6 flex justify-between items-center">
          <motion.h1 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-3xl font-bold text-indigo-600"
          >
            Cerse
          </motion.h1>
          <p className="text-lg text-gray-600">AI IPC Prediction</p>
        </div>
      </header>
      
      <nav className="bg-indigo-600 text-white">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex space-x-4">
            {/* IPC Pulse Button */}
            <button 
              onClick={() => setActiveTab('pulse')}
              className={`py-4 px-6 ${activeTab === 'pulse' ? 'bg-indigo-700' : ''}`}
            >
              IPC Pulse
            </button>
            {/* Will add back in future if there is time */}
            {/* <button 
              onClick={() => setActiveTab('navigator')}
              className={`py-4 px-6 ${activeTab === 'navigator' ? 'bg-indigo-700' : ''}`}
            >
              IPC Navigator (Coming Soon)
            </button> */}
          </div>
        </div>
      </nav>
      
      <main className="max-w-7xl mx-auto px-4 py-8">
        {activeTab === 'pulse' && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="bg-white rounded-lg shadow-lg p-6"
          >
            {countriesLoading ? (
              <p className="text-center text-gray-600">Loading countries...</p>
            ) : error ? (
              <p className="text-red-500 text-center">{error}</p>
            ) : (
              <div className="flex flex-col md:flex-row space-y-4 md:space-y-0 md:space-x-4 mb-6">
                <Select
                  options={countries.map(c => ({ value: c, label: c }))}
                  value={{ value: selectedCountry, label: selectedCountry }}
                  onChange={option => setSelectedCountry(option.value)}
                  placeholder="Select Country"
                  className="flex-1"
                />
                {levels.length > 0 && (
                  <Select
                    options={levels.map(l => ({ value: l, label: l }))}
                    value={{ value: selectedLevel, label: selectedLevel }}
                    onChange={option => setSelectedLevel(option.value)}
                    placeholder="Select Region"
                    className="flex-1"
                  />
                )}
                {areas.length > 0 && (
                  <Select
                    options={areas.map(a => ({ value: a, label: a }))}
                    value={{ value: selectedArea, label: selectedArea }}
                    onChange={option => setSelectedArea(option.value)}
                    placeholder="Select Area"
                    className="flex-1"
                  />
                )}
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={loadGraph}
                  disabled={!selectedCountry || loading}
                  className="bg-indigo-600 text-white px-6 py-2 rounded-md disabled:opacity-50"
                >
                  {loading ? 'Loading...' : 'Load Graph'}
                </motion.button>
              </div>
            )}
            {error && <p className="text-red-500 mb-4">{error}</p>}
            {graphData.length > 0 && <IpcChart data={graphData} />}
          </motion.div>
        )}
        {activeTab === 'navigator' && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="bg-white rounded-lg shadow-lg p-6 text-center"
          >
            <h2 className="text-2xl font-bold mb-4">IPC Navigator</h2>
            <p className="text-gray-600">Visualize predictions on an interactive map. Coming soon!</p>
          </motion.div>
        )}
      </main>
    </div>
  );
}

export default App;
