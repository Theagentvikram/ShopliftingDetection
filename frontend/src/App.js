import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Navbar from './components/Navbar';
import LiveDetection from './components/LiveDetection';
import VideoUpload from './components/VideoUpload';
import Training from './components/Training';
import Dashboard from './components/Dashboard';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <div className="App">
          <Navbar />
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/live" element={<LiveDetection />} />
            <Route path="/upload" element={<VideoUpload />} />
            <Route path="/training" element={<Training />} />
          </Routes>
        </div>
      </Router>
    </ThemeProvider>
  );
}

export default App;
