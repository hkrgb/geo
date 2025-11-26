import React from 'react';

const LoadingSpinner: React.FC = () => {
  return (
    <div className="flex flex-col items-center justify-center h-64 space-y-4">
      <div className="relative w-20 h-20">
        <div className="absolute top-0 left-0 w-full h-full border-4 border-blue-400 border-opacity-20 rounded-full"></div>
        <div className="absolute top-0 left-0 w-full h-full border-4 border-blue-500 rounded-full animate-spin border-t-transparent"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-blue-300">
          <i className="fas fa-satellite-dish text-xl animate-pulse"></i>
        </div>
      </div>
      <p className="text-blue-200 animate-pulse font-medium">正如火如荼分析地理资讯...</p>
    </div>
  );
};

export default LoadingSpinner;