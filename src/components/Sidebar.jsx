import React from 'react';
import { NavLink } from 'react-router-dom';
import { RiDashboardLine, RiLineChartLine, RiAlertLine, RiSettings4Line, RiVideoLine } from 'react-icons/ri';

function Sidebar() {
  const navItems = [
    { path: '/', icon: RiDashboardLine, text: 'Dashboard' },
    { path: '/analytics', icon: RiLineChartLine, text: 'Analytics' },
    { path: '/alerts', icon: RiAlertLine, text: 'Alerts' },
    { path: '/monitoring', icon: RiVideoLine, text: 'Monitoring' },
    { path: '/settings', icon: RiSettings4Line, text: 'Settings' },
  ];

  return (
    <div className="w-64 bg-gray-800 text-white p-4">
      <div className="text-xl font-bold mb-8 p-4">
        Retail Theft Detection
      </div>
      <nav>
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              `flex items-center space-x-3 p-3 rounded-lg transition-colors ${
                isActive
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-300 hover:bg-gray-700'
              }`
            }
          >
            <item.icon className="text-xl" />
            <span>{item.text}</span>
          </NavLink>
        ))}
      </nav>
    </div>
  );
}

export default Sidebar