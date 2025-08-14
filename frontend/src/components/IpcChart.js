import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Label } from 'recharts';
import { motion } from 'framer-motion';

function IpcChart({ data }) {
  // Use the chart-ready data from backend
  const chartData = data.map(d => d.chart_data);
  
  // Separate historical and predicted data for different line styling
  const historicalData = chartData.filter(d => !d.is_predicted);
  const predictedData = chartData.filter(d => d.is_predicted);
  
  // Create complete dataset with null values for predicted periods in historical data
  // and null values for historical periods in predicted data
  const completeData = chartData.map(d => ({
    year: d.year,
    // Historical data (null for predicted years)
    hist_phase1: !d.is_predicted ? d.phase1 : null,
    hist_phase2: !d.is_predicted ? d.phase2 : null,
    hist_phase3: !d.is_predicted ? d.phase3 : null,
    hist_phase4: !d.is_predicted ? d.phase4 : null,
    hist_phase5: !d.is_predicted ? d.phase5 : null,
    // Predicted data (null for historical years, but include connection point)
    pred_phase1: d.is_predicted ? d.phase1 : null,
    pred_phase2: d.is_predicted ? d.phase2 : null,
    pred_phase3: d.is_predicted ? d.phase3 : null,
    pred_phase4: d.is_predicted ? d.phase4 : null,
    pred_phase5: d.is_predicted ? d.phase5 : null,
    is_predicted: d.is_predicted
  }));
  
  // Add connection points for smooth transition
  if (historicalData.length > 0 && predictedData.length > 0) {
    const lastHistorical = historicalData[historicalData.length - 1];
    const connectionIndex = completeData.findIndex(d => d.year === lastHistorical.year);
    if (connectionIndex >= 0) {
      completeData[connectionIndex].pred_phase1 = lastHistorical.phase1;
      completeData[connectionIndex].pred_phase2 = lastHistorical.phase2;
      completeData[connectionIndex].pred_phase3 = lastHistorical.phase3;
      completeData[connectionIndex].pred_phase4 = lastHistorical.phase4;
      completeData[connectionIndex].pred_phase5 = lastHistorical.phase5;
    }
  }

  // Check if this specific graph has both historical and predicted data
  const hasHistoricalData = historicalData.length > 0;
  const hasPredictedData = predictedData.length > 0;
  const showGuide = hasHistoricalData && hasPredictedData;

  // Custom legend to only show historical phases. Do not display predicted phases on the legend of the IPC Pulse Graph to prevent regression.
  const renderLegend = (props) => {
    const { payload } = props;
    return (
      <ul style={{ padding: '20px 0 0 0', fontSize: '14px', textAlign: 'center', margin: '0 auto', listStyleType: 'none' }}>
        {payload.filter(item => item.dataKey && item.dataKey.startsWith('hist_')).map((entry, index) => (
          <li key={`item-${index}`} style={{ display: 'inline-block', margin: '0 10px' }}>
            <svg width="14" height="14" viewBox="0 0 32 32" style={{ display: 'inline-block', verticalAlign: 'middle', marginRight: '4px' }}>
              <circle cx="16" cy="16" r="10" fill={entry.color} stroke="none" />
            </svg>
            <span style={{ color: entry.color }}>{entry.value}</span>
          </li>
        ))}
      </ul>
    );
  };

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      className="bg-white rounded-lg shadow-xl p-6"
    >
      <h2 className="text-2xl font-bold text-center mb-6 text-indigo-800">
        IPC Pulse
      </h2>
      {showGuide && (
        <p className="text-sm text-gray-600 mb-4 text-center italic">
          Solid lines: Historical | Dashed lines: AI Predictions
        </p>
      )}
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={completeData} margin={{ top: 20, right: 30, left: 60, bottom: 60 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.5} />
          <XAxis dataKey="year" tick={{ fill: '#6b7280' }}>
            <Label value="Year" position="insideBottom" offset={-5} style={{ textAnchor: 'middle', fill: '#4b5563' }} />
          </XAxis>
          <YAxis domain={[0, 100]} tick={{ fill: '#6b7280' }}>
            <Label 
              value="Population Affected (%)" 
              angle={-90} 
              position="insideLeft" 
              style={{ textAnchor: 'middle', fill: '#4b5563' }} 
            />
          </YAxis>
          <Tooltip 
            wrapperStyle={{ background: 'white', border: '1px solid #e5e7eb', borderRadius: '8px', padding: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}
            content={(props) => {
              const { active, payload, label } = props;
              if (!active || !payload || !payload.length) return null;

              // Get the data for this point
              const data = payload[0].payload;
              const isPredicted = data.is_predicted;

              // Use hist_ for historical, pred_ for predicted
              const phasePrefix = isPredicted ? 'pred_phase' : 'hist_phase';
              const title = `${label} (${isPredicted ? 'Predicted' : 'Historical'})`;

              return (
                <div className="bg-white p-4 border rounded shadow">
                  <p className="font-bold">{title}</p>
                  <ul>
                    <li style={{color: '#22c55e'}}>Phase 1: {data[`${phasePrefix}1`]}%</li>
                    <li style={{color: '#eab308'}}>Phase 2: {data[`${phasePrefix}2`]}%</li>
                    <li style={{color: '#f97316'}}>Phase 3: {data[`${phasePrefix}3`]}%</li>
                    <li style={{color: '#ef4444'}}>Phase 4: {data[`${phasePrefix}4`]}%</li>
                    <li style={{color: '#991b1b'}}>Phase 5: {data[`${phasePrefix}5`]}%</li>
                  </ul>
                </div>
              );
            }}
          />
          {/* Replace with custom legend */}
          <Legend content={renderLegend} />
          
          {/* Historical lines with animation */}
          <Line type="monotone" dataKey="hist_phase1" name="Phase 1 (Minimal)" stroke="#22c55e" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
          <Line type="monotone" dataKey="hist_phase2" name="Phase 2 (Stressed)" stroke="#eab308" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
          <Line type="monotone" dataKey="hist_phase3" name="Phase 3 (Crisis)" stroke="#f97316" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
          <Line type="monotone" dataKey="hist_phase4" name="Phase 4 (Emergency)" stroke="#ef4444" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
          <Line type="monotone" dataKey="hist_phase5" name="Phase 5 (Catastrophe)" stroke="#991b1b" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
          
          {/* Predicted lines - Do not display predicted phases in the legend (use legendType="none" to prevent regression) */}
          <Line type="monotone" dataKey="pred_phase1" name="" stroke="#22c55e" strokeWidth={3} strokeDasharray="5 5" strokeOpacity={0.7} dot={{ r: 4 }} activeDot={{ r: 6 }} legendType="none" />
          <Line type="monotone" dataKey="pred_phase2" name="" stroke="#eab308" strokeWidth={3} strokeDasharray="5 5" strokeOpacity={0.7} dot={{ r: 4 }} activeDot={{ r: 6 }} legendType="none" />
          <Line type="monotone" dataKey="pred_phase3" name="" stroke="#f97316" strokeWidth={3} strokeDasharray="5 5" strokeOpacity={0.7} dot={{ r: 4 }} activeDot={{ r: 6 }} legendType="none" />
          <Line type="monotone" dataKey="pred_phase4" name="" stroke="#ef4444" strokeWidth={3} strokeDasharray="5 5" strokeOpacity={0.7} dot={{ r: 4 }} activeDot={{ r: 6 }} legendType="none" />
          <Line type="monotone" dataKey="pred_phase5" name="" stroke="#991b1b" strokeWidth={3} strokeDasharray="5 5" strokeOpacity={0.7} dot={{ r: 4 }} activeDot={{ r: 6 }} legendType="none" />
        </LineChart>
      </ResponsiveContainer>
    </motion.div>
  );
}

export default IpcChart;
