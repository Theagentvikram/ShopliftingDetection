import React from 'react';

function Alerts() {
  const alerts = [
    {
      id: 1,
      store: 'Store A',
      camera: 'Camera 3',
      type: 'Suspicious Activity',
      time: '2 minutes ago',
      status: 'Active'
    },
    {
      id: 2,
      store: 'Store B',
      camera: 'Camera 1',
      type: 'Theft Attempt',
      time: '15 minutes ago',
      status: 'Resolved'
    },
    {
      id: 3,
      store: 'Store C',
      camera: 'Camera 5',
      type: 'Unknown Person',
      time: '1 hour ago',
      status: 'Active'
    }
  ];

  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold mb-8">Alerts</h1>
      
      <div className="bg-white rounded-lg shadow">
        <div className="p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-bold">Recent Alerts</h2>
            <button className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600">
              Clear All
            </button>
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead>
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Store
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Camera
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Type
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Time
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Action
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {alerts.map((alert) => (
                  <tr key={alert.id}>
                    <td className="px-6 py-4 whitespace-nowrap">{alert.store}</td>
                    <td className="px-6 py-4 whitespace-nowrap">{alert.camera}</td>
                    <td className="px-6 py-4 whitespace-nowrap">{alert.type}</td>
                    <td className="px-6 py-4 whitespace-nowrap">{alert.time}</td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                        alert.status === 'Active' ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'
                      }`}>
                        {alert.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <button className="text-blue-600 hover:text-blue-900">View Details</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Alerts