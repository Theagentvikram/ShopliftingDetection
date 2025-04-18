import React from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

function Analytics() {
  const data = {
    labels: ['Store A', 'Store B', 'Store C', 'Store D', 'Store E'],
    datasets: [
      {
        label: 'Theft Attempts',
        data: [65, 59, 80, 81, 56],
        backgroundColor: 'rgba(54, 162, 235, 0.5)',
      },
      {
        label: 'Prevented Thefts',
        data: [45, 49, 60, 71, 46],
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
      }
    ]
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Theft Prevention Analysis by Store'
      }
    }
  };

  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold mb-8">Analytics</h1>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-bold mb-4">Store Performance</h2>
          <Bar options={options} data={data} />
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-bold mb-4">Key Metrics</h2>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span>Detection Accuracy</span>
              <span className="font-bold">95%</span>
            </div>
            <div className="flex justify-between items-center">
              <span>Average Response Time</span>
              <span className="font-bold">2.3 seconds</span>
            </div>
            <div className="flex justify-between items-center">
              <span>False Positive Rate</span>
              <span className="font-bold">3%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Analytics