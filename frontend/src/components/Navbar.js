import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box
} from '@mui/material';
import {
  Videocam,
  CloudUpload,
  School,
  Dashboard
} from '@mui/icons-material';
import { Link as RouterLink } from 'react-router-dom';

const Navbar = () => {
  return (
    <AppBar position="static">
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          Shoplifting Detection
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            color="inherit"
            component={RouterLink}
            to="/"
            startIcon={<Dashboard />}
          >
            Dashboard
          </Button>
          
          <Button
            color="inherit"
            component={RouterLink}
            to="/live"
            startIcon={<Videocam />}
          >
            Live Detection
          </Button>
          
          <Button
            color="inherit"
            component={RouterLink}
            to="/upload"
            startIcon={<CloudUpload />}
          >
            Upload Video
          </Button>
          
          <Button
            color="inherit"
            component={RouterLink}
            to="/training"
            startIcon={<School />}
          >
            Training
          </Button>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;
