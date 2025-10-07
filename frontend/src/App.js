import React from 'react';

// Import your components
import FrontPage from './Components/FrontPage/FrontPage';
import Navbar from './Components/Navbar/Navbar';
import Home from './Components/Home/Home';
import Upload from './Components/Upload/Upload';
import Title from './Components/Title/Title';

const App = () => {
  return (
    <div>
      {/* Navbar Component */}
      <Navbar />

      {/* Home Component */}
      <div className="homerouter">
        <Home />
      </div>

      {/* Upload Section */}
      <div className="uploadrouter">
        <div className="container">
          {/* Title and Upload Components */}
          <Title subTitle="Re-Search Bot" title="Attach Documents to get Started" />
          <Upload />
        </div>
      </div>

      <br />
    </div>
  );
};

export default App;
