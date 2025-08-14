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
  const [selectedCountry, setSelectedCountry] = useState('');
  const [selectedLevel, setSelectedLevel] = useState('');
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
    }
  }, [selectedCountry]);

  const loadGraph = async () => {
    setLoading(true);
    setError('');
    try {
      let params = `country=${encodeURIComponent(selectedCountry)}`;
      if (selectedLevel) params += `&level1=${encodeURIComponent(selectedLevel)}`;
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
            className="text-5xl font-extrabold text-white tracking-wide"
          >
            {/* company name */}
            Ceres
          </motion.h1>
          <p className="text-2xl text-indigo-100 font-medium">AI IPC Prediction Platform</p>
        </div>
      </header>
      
      <nav className="bg-indigo-800 text-white">
        <div className="px-6">
          <div className="flex space-x-4">
            {/* IPC Pulse Button */}
            <button 
              onClick={() => setActiveTab('pulse')}
              className={`py-5 px-8 text-lg ${activeTab === 'pulse' ? 'bg-indigo-700' : ''}`}
            >
              IPC Pulse
            </button>
            {/* How It Works Button */}
            <button 
              onClick={() => setActiveTab('how-it-works')}
              className={`py-4 px-6 ${activeTab === 'how-it-works' ? 'bg-indigo-700' : ''}`}
            >
              How It Works
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
            className="bg-white rounded-lg shadow-lg p-8" // Increased padding
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
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={loadGraph}
                  disabled={!selectedCountry || loading}
                  className="bg-indigo-600 text-white px-8 py-3 rounded-md disabled:opacity-50 text-lg"
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
                  setGraphData([]);
                  setLevels([]);
                }} 
              />
            ) : (
              <motion.div 
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
                className="bg-white rounded-lg shadow-xl p-8" // Increased padding
                style={{ height: 600 }} // Increased height to match larger graph
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
        
        {activeTab === 'how-it-works' && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="space-y-8"
          >
            {/* Hero Section */}
            <div className="bg-white rounded-lg shadow-lg p-8">
              <motion.h2 
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="text-3xl font-bold text-indigo-800 mb-4"
              >
                How Ceres AI IPC Prediction Platform Works
              </motion.h2>
              <motion.p 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2 }}
                className="text-lg text-gray-700 leading-relaxed"
              >
                Our AI-powered platform combines historical IPC (Integrated Food Security Phase Classification) data 
                with cutting-edge satellite imagery embeddings from Google's AlphaEarth model to predict future food security phases, 
                helping organizations make informed decisions about humanitarian interventions.
              </motion.p>
            </div>

            {/* Overall System Flow */}
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="bg-white rounded-lg shadow-lg p-8"
            >
              <h3 className="text-2xl font-bold text-indigo-700 mb-6 flex items-center">
                <svg className="w-8 h-8 mr-3 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2-2v10a2 2 0 002 2h2a2 2 0 002-2z" />
                </svg>
                Overall System Flow
              </h3>
              <p className="text-gray-700 mb-6">
                Our system is a sophisticated data pipeline that enriches local IPC datasets with satellite imagery embeddings, 
                combining three main components: local data sources, geospatial libraries, and remote processing services.
              </p>
              
              <div className="grid md:grid-cols-5 gap-4">
                <div className="text-center">
                  <div className="bg-blue-100 rounded-full p-4 w-16 h-16 mx-auto mb-3 flex items-center justify-center">
                    <span className="text-2xl">📝</span>
                  </div>
                  <h4 className="font-semibold text-blue-700 mb-2">1. Data Source</h4>
                  <p className="text-sm text-gray-600">Read local IPC dataset and extract countries and years</p>
                </div>
                <div className="text-center">
                  <div className="bg-green-100 rounded-full p-4 w-16 h-16 mx-auto mb-3 flex items-center justify-center">
                    <span className="text-2xl">🗺️</span>
                  </div>
                  <h4 className="font-semibold text-green-700 mb-2">2. Boundaries</h4>
                  <p className="text-sm text-gray-600">Download administrative boundaries using pygadm library</p>
                </div>
                <div className="text-center">
                  <div className="bg-yellow-100 rounded-full p-4 w-16 h-16 mx-auto mb-3 flex items-center justify-center">
                    <span className="text-2xl">🔄</span>
                  </div>
                  <h4 className="font-semibold text-yellow-700 mb-2">3. Translation</h4>
                  <p className="text-sm text-gray-600">Convert local maps to Google Earth Engine format</p>
                </div>
                <div className="text-center">
                  <div className="bg-purple-100 rounded-full p-4 w-16 h-16 mx-auto mb-3 flex items-center justify-center">
                    <span className="text-2xl">🤖</span>
                  </div>
                  <h4 className="font-semibold text-purple-700 mb-2">4. Processing</h4>
                  <p className="text-sm text-gray-600">Extract satellite embeddings using AlphaEarth model</p>
                </div>
                <div className="text-center">
                  <div className="bg-indigo-100 rounded-full p-4 w-16 h-16 mx-auto mb-3 flex items-center justify-center">
                    <span className="text-2xl">📦</span>
                  </div>
                  <h4 className="font-semibold text-indigo-700 mb-2">5. Output</h4>
                  <p className="text-sm text-gray-600">Generate enriched dataset with satellite embeddings</p>
                </div>
              </div>
            </motion.div>

            {/* AlphaEarth Section */}
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="bg-gradient-to-r from-purple-50 to-indigo-50 rounded-lg shadow-lg p-8"
            >
              <h3 className="text-2xl font-bold text-indigo-700 mb-6 flex items-center">
                <svg className="w-8 h-8 mr-3 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                AlphaEarth: The Virtual Satellite
              </h3>
              <div className="grid md:grid-cols-2 gap-8">
                <div>
                  <p className="text-gray-700 mb-4">
                    AlphaEarth Foundations is Google DeepMind's revolutionary AI model that acts as a "virtual satellite" 
                    to map our planet in unprecedented detail. It synthesizes vast amounts of Earth observation data into 
                    64-dimensional numerical vectors called embeddings.
                  </p>
                  <h4 className="text-lg font-semibold mb-3 text-purple-700">Key Innovation</h4>
                  <p className="text-gray-700 mb-4">
                    Each 10x10 meter pixel on Earth gets a unique 64-number "fingerprint" that captures comprehensive 
                    information about that location for a given year. These embeddings are analysis-ready and consistent, 
                    even in areas with missing data or cloud cover.
                  </p>
                </div>
                <div>
                  <h4 className="text-lg font-semibold mb-3 text-purple-700">Data Sources Integrated</h4>
                  <ul className="space-y-2 text-gray-700">
                    <li className="flex items-start">
                      <svg className="w-5 h-5 text-green-500 mr-2 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      Optical satellite imagery (Sentinel-2, Landsat)
                    </li>
                    <li className="flex items-start">
                      <svg className="w-5 h-5 text-green-500 mr-2 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      Radar data (Sentinel-1) - sees through clouds
                    </li>
                    <li className="flex items-start">
                      <svg className="w-5 h-5 text-green-500 mr-2 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      3D laser mapping (LiDAR) for terrain models
                    </li>
                    <li className="flex items-start">
                      <svg className="w-5 h-5 text-green-500 mr-2 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      Climate simulations and meteorological data
                    </li>
                    <li className="flex items-start">
                      <svg className="w-5 h-5 text-green-500 mr-2 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      Additional geospatial information sources
                    </li>
                  </ul>
                </div>
              </div>
            </motion.div>

            <div className="grid md:grid-cols-2 gap-8">
              {/* What is IPC Section - Enhanced */}
              <motion.div 
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.5 }}
                className="bg-white rounded-lg shadow-lg p-6"
              >
                <h3 className="text-2xl font-bold text-indigo-700 mb-4 flex items-center">
                  <svg className="w-8 h-8 mr-3 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  Understanding IPC Classifications
                </h3>
                <div className="space-y-4">
                  <p className="text-gray-700">
                    The Integrated Food Security Phase Classification (IPC) is a global standard for classifying 
                    food insecurity severity. An area is classified in a phase when at least 20% of its population 
                    meets the criteria for that phase or higher.
                  </p>
                  <div className="space-y-3">
                    <div className="border-l-4 border-green-500 pl-4 py-2 bg-green-50">
                      <div className="flex items-center space-x-3 mb-1">
                        <div className="w-4 h-4 bg-green-500 rounded-full"></div>
                        <span className="font-bold text-green-800">Phase 1: Minimal</span>
                      </div>
                      <p className="text-sm text-gray-700">Households meet food and basic needs without unusual methods. Death rate &lt; 0.5/10,000/day.</p>
                    </div>
                    <div className="border-l-4 border-yellow-400 pl-4 py-2 bg-yellow-50">
                      <div className="flex items-center space-x-3 mb-1">
                        <div className="w-4 h-4 bg-yellow-400 rounded-full"></div>
                        <span className="font-bold text-yellow-800">Phase 2: Stressed</span>
                      </div>
                      <p className="text-sm text-gray-700">Just enough food, but can't afford non-food essentials. Some livelihood stress. Death rate &lt; 0.5/10,000/day.</p>
                    </div>
                    <div className="border-l-4 border-orange-500 pl-4 py-2 bg-orange-50">
                      <div className="flex items-center space-x-3 mb-1">
                        <div className="w-4 h-4 bg-orange-500 rounded-full"></div>
                        <span className="font-bold text-orange-800">Phase 3: Crisis</span>
                      </div>
                      <p className="text-sm text-gray-700">Food gaps or extreme coping (selling livestock). High malnutrition. Death rate 0.5-0.99/10,000/day.</p>
                    </div>
                    <div className="border-l-4 border-red-500 pl-4 py-2 bg-red-50">
                      <div className="flex items-center space-x-3 mb-1">
                        <div className="w-4 h-4 bg-red-500 rounded-full"></div>
                        <span className="font-bold text-red-800">Phase 4: Emergency</span>
                      </div>
                      <p className="text-sm text-gray-700">Large food gaps, very high malnutrition, emergency strategies. Death rate 1-1.99/10,000/day.</p>
                    </div>
                    <div className="border-l-4 border-red-900 pl-4 py-2 bg-red-100">
                      <div className="flex items-center space-x-3 mb-1">
                        <div className="w-4 h-4 bg-red-900 rounded-full"></div>
                        <span className="font-bold text-red-900">Phase 5: Famine</span>
                      </div>
                      <p className="text-sm text-gray-700">Complete lack of food, starvation, widespread death. Malnutrition &gt;30%. Death rate ≥2/10,000/day.</p>
                    </div>
                  </div>
                </div>
              </motion.div>

              {/* AI Predictions Section */}
              <motion.div 
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.6 }}
                className="bg-white rounded-lg shadow-lg p-6"
              >
                <h3 className="text-2xl font-bold text-indigo-700 mb-4 flex items-center">
                  <svg className="w-8 h-8 mr-3 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                  AI-Powered Predictions
                </h3>
                <div className="space-y-4">
                  <p className="text-gray-700">
                    Our advanced AI algorithms combine historical IPC trends with AlphaEarth satellite embeddings 
                    to predict future food security phases for 2026, enabling proactive humanitarian planning.
                  </p>
                  <h4 className="text-lg font-semibold text-indigo-600">Prediction Methodology</h4>
                  <ul className="space-y-2 text-gray-700">
                    <li className="flex items-start">
                      <svg className="w-5 h-5 text-green-500 mr-2 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      Multi-year historical IPC data analysis
                    </li>
                    <li className="flex items-start">
                      <svg className="w-5 h-5 text-green-500 mr-2 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      64-dimensional satellite embedding integration
                    </li>
                    <li className="flex items-start">
                      <svg className="w-5 h-5 text-green-500 mr-2 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      Seasonal variation pattern recognition
                    </li>
                    <li className="flex items-start">
                      <svg className="w-5 h-5 text-green-500 mr-2 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      Crisis escalation probability modeling
                    </li>
                    <li className="flex items-start">
                      <svg className="w-5 h-5 text-green-500 mr-2 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      Regional and country-level forecasting
                    </li>
                  </ul>
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <h5 className="font-semibold text-blue-800 mb-2">Early Warning Capability</h5>
                    <p className="text-sm text-blue-700">
                      By combining satellite data with historical patterns, our system can identify areas 
                      at risk of food security deterioration months before traditional methods.
                    </p>
                  </div>
                </div>
              </motion.div>
            </div>

            {/* How to Use Section */}
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7 }}
              className="bg-white rounded-lg shadow-lg p-8"
            >
              <h3 className="text-2xl font-bold text-indigo-700 mb-6 flex items-center">
                <svg className="w-8 h-8 mr-3 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                </svg>
                How to Use the Platform
              </h3>
              <div className="grid md:grid-cols-3 gap-6">
                <div className="text-center">
                  <div className="bg-indigo-100 rounded-full p-4 w-16 h-16 mx-auto mb-4 flex items-center justify-center">
                    <span className="text-2xl font-bold text-indigo-600">1</span>
                  </div>
                  <h4 className="text-lg font-semibold mb-2">Select Country</h4>
                  <p className="text-gray-600">Choose a country from the dropdown menu to begin your analysis.</p>
                </div>
                <div className="text-center">
                  <div className="bg-indigo-100 rounded-full p-4 w-16 h-16 mx-auto mb-4 flex items-center justify-center">
                    <span className="text-2xl font-bold text-indigo-600">2</span>
                  </div>
                  <h4 className="text-lg font-semibold mb-2">Refine Selection</h4>
                  <p className="text-gray-600">Optionally select specific regions and areas for more detailed insights.</p>
                </div>
                <div className="text-center">
                  <div className="bg-indigo-100 rounded-full p-4 w-16 h-16 mx-auto mb-4 flex items-center justify-center">
                    <span className="text-2xl font-bold text-indigo-600">3</span>
                  </div>
                  <h4 className="text-lg font-semibold mb-2">View Results</h4>
                  <p className="text-gray-600">Click "Load Graph" to see historical trends and AI predictions.</p>
                </div>
              </div>
            </motion.div>

            {/* Understanding the Charts Section */}
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8 }}
              className="bg-white rounded-lg shadow-lg p-8"
            >
              <h3 className="text-2xl font-bold text-indigo-700 mb-6 flex items-center">
                <svg className="w-8 h-8 mr-3 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                Understanding the Charts
              </h3>
              <div className="grid md:grid-cols-2 gap-8">
                <div>
                  <h4 className="text-lg font-semibold mb-3 text-indigo-600">Historical Data (Solid Lines)</h4>
                  <ul className="space-y-2 text-gray-700">
                    <li>• Shows actual IPC phase data from past years</li>
                    <li>• Each color represents a different IPC phase</li>
                    <li>• Line thickness indicates population percentages</li>
                    <li>• Trends help identify seasonal patterns</li>
                    <li>• Based on official IPC assessments and reports</li>
                  </ul>
                </div>
                <div>
                  <h4 className="text-lg font-semibold mb-3 text-indigo-600">AI Predictions (Dashed Lines)</h4>
                  <ul className="space-y-2 text-gray-700">
                    <li>• Forecasts for 2026 based on historical patterns</li>
                    <li>• Dashed lines distinguish predictions from facts</li>
                    <li>• Incorporates satellite embedding data</li>
                    <li>• Considers crisis escalation probabilities</li>
                    <li>• Helps plan early intervention strategies</li>
                  </ul>
                </div>
              </div>
            </motion.div>

            {/* Use Cases Section */}
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.9 }}
              className="bg-gradient-to-r from-indigo-50 to-blue-50 rounded-lg shadow-lg p-8"
            >
              <h3 className="text-2xl font-bold text-indigo-700 mb-6 text-center">Who Can Benefit from This Platform?</h3>
              <div className="grid md:grid-cols-4 gap-6">
                <div className="text-center">
                  <div className="bg-white rounded-lg p-4 shadow-md mb-3">
                    <svg className="w-12 h-12 mx-auto text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                    </svg>
                  </div>
                  <h4 className="font-semibold text-indigo-700">Humanitarian Organizations</h4>
                  <p className="text-sm text-gray-600 mt-2">Plan interventions and resource allocation based on predictive insights</p>
                </div>
                <div className="text-center">
                  <div className="bg-white rounded-lg p-4 shadow-md mb-3">
                    <svg className="w-12 h-12 mx-auto text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                    </svg>
                  </div>
                  <h4 className="font-semibold text-indigo-700">Government Agencies</h4>
                  <p className="text-sm text-gray-600 mt-2">Inform policy decisions and early warning systems with AI predictions</p>
                </div>
                <div className="text-center">
                  <div className="bg-white rounded-lg p-4 shadow-md mb-3">
                    <svg className="w-12 h-12 mx-auto text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <h4 className="font-semibold text-indigo-700">Donors & Funders</h4>
                  <p className="text-sm text-gray-600 mt-2">Make evidence-based funding decisions using satellite-enhanced data</p>
                </div>
                <div className="text-center">
                  <div className="bg-white rounded-lg p-4 shadow-md mb-3">
                    <svg className="w-12 h-12 mx-auto text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                  </div>
                  <h4 className="font-semibold text-indigo-700">Researchers</h4>
                  <p className="text-sm text-gray-600 mt-2">Study food security trends with cutting-edge satellite embedding data</p>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
        
        {activeTab === 'navigator' && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
                          className="bg-white rounded-lg shadow-lg p-8" // Increased padding
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
