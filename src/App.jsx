import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Analytics from './pages/Analytics';
import Alerts from './pages/Alerts';
import Settings from './pages/Settings';
import Monitoring from './pages/Monitoring';
import StatusPage from './pages/RecoveryPage';

// Error boundary component
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { 
      hasError: false,
      error: null,
      errorInfo: null
    };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("Component Error:", error, errorInfo);
    this.setState({ errorInfo });
  }

  render() {
    if (this.state.hasError) {
      return <StatusPage error={this.state.error} />;
    }
    return this.props.children;
  }
}

function App() {
  const [analysisResults, setAnalysisResults] = useState(null);

  const handleAnalysisComplete = (results) => {
    setAnalysisResults(results);
  };

  return (
    <Router>
      <ErrorBoundary>
        <div className="flex h-screen bg-gray-100">
          <Sidebar />
          <div className="flex-1 overflow-auto">
            <Routes>
              <Route path="/" element={<Navigate to="/monitoring" replace />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/alerts" element={<Alerts />} />
              <Route 
                path="/monitoring" 
                element={
                  <Monitoring 
                    onAnalysisComplete={handleAnalysisComplete}
                  />
                } 
              />
              <Route path="/settings" element={<Settings />} />
              <Route 
                path="/status" 
                element={
                  <StatusPage 
                    analysisResults={analysisResults}
                  />
                } 
              />
            </Routes>
          </div>
        </div>
      </ErrorBoundary>
    </Router>
  );
}

export default App;