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
      <header className="bg-gradient-to-r from-indigo-600 to-indigo-800 shadow-lg">
        <div className="px-6 py-6 flex justify-between items-center">
          <motion.h1 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="text-4xl font-extrabold text-white tracking-wide"
          >
            {/* company name */}
            Ceres
          </motion.h1>
          <p className="text-xl text-indigo-100 font-medium">AI IPC Prediction Platform</p>
        </div>
      </header>
      
      <nav className="bg-indigo-800 text-white">
        <div className="px-6">
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
      
      <main className="px-6 py-8">
        {activeTab === 'pulse' && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="bg-white rounded-lg shadow-lg p-4" // Reduced padding
          >
            {countriesLoading ? (
              <p className="text-center text-gray-600">Loading countries...</p>
            ) : error ? (
              <p className="text-red-500 text-center">{error}</p>
            ) : (
              <div className="flex flex-col md:flex-row space-y-4 md:space-y-0 md:space-x-4 mb-6">
                <div className="flex-1">
                  <label htmlFor="country-select" className="block text-sm font-medium text-gray-700 mb-1">Select Country</label>
                  <Select
                    id="country-select"
                    options={countries.map(c => ({ value: c, label: c }))}
                    value={{ value: selectedCountry, label: selectedCountry }}
                    onChange={option => setSelectedCountry(option.value)}
                    placeholder="Select Country"
                    className="flex-1"
                  />
                </div>
                {levels.length > 0 && (
                  <div className="flex-1">
                    <label htmlFor="region-select" className="block text-sm font-medium text-gray-700 mb-1">Select Region</label>
                    <Select
                      id="region-select"
                      options={levels.map(l => ({ value: l, label: l }))}
                      value={{ value: selectedLevel, label: selectedLevel }}
                      onChange={option => setSelectedLevel(option.value)}
                      placeholder="Select Region"
                      className="flex-1"
                    />
                  </div>
                )}
                {areas.length > 0 && (
                  <div className="flex-1">
                    <label htmlFor="area-select" className="block text-sm font-medium text-gray-700 mb-1">Select Area</label>
                    <Select
                      id="area-select"
                      options={areas.map(a => ({ value: a, label: a }))}
                      value={{ value: selectedArea, label: selectedArea }}
                      onChange={option => setSelectedArea(option.value)}
                      placeholder="Select Area"
                      className="flex-1"
                    />
                  </div>
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
            {graphData.length > 0 ? (
              <IpcChart 
                data={graphData} 
                onReset={() => {
                  setSelectedCountry('');
                  setSelectedLevel('');
                  setSelectedArea('');
                  setGraphData([]);
                  setLevels([]);
                  setAreas([]);
                }} 
              />
            ) : (
              <motion.div 
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
                className="bg-white rounded-lg shadow-xl p-4" // Reduced padding
                style={{ height: 500 }} // Approximate height to match graph
              >
                <h2 className="text-2xl font-bold text-center mb-6 text-indigo-800">
                  IPC Pulse
                </h2>
                <div className="flex items-center justify-center h-80 text-gray-500 text-lg">
                  <p>Select a country and click "Load Graph" to view the IPC data.</p>
                </div>
              </motion.div>
            )}
          </motion.div>
        )}
        {activeTab === 'navigator' && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="bg-white rounded-lg shadow-lg p-4" // Reduced padding
            style={{ height: 500 }}
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
