import React from 'react';
import { WeatherInfo } from '../types';

interface WeatherCardProps {
  weather: WeatherInfo;
}

const WeatherCard: React.FC<WeatherCardProps> = ({ weather }) => {
  // Simple icon mapping based on condition text
  const getIcon = (condition: string) => {
    if (condition.includes('雨')) return 'fa-cloud-rain';
    if (condition.includes('云') || condition.includes('雲')) return 'fa-cloud';
    if (condition.includes('晴')) return 'fa-sun';
    if (condition.includes('雪')) return 'fa-snowflake';
    if (condition.includes('雷')) return 'fa-bolt';
    return 'fa-cloud-sun';
  };

  return (
    <div className="glass-panel p-6 rounded-2xl mb-6 text-white relative overflow-hidden group">
      <div className="absolute -right-10 -top-10 w-40 h-40 bg-blue-500 rounded-full mix-blend-overlay filter blur-3xl opacity-20 group-hover:opacity-30 transition-opacity"></div>
      
      <div className="flex items-center justify-between relative z-10">
        <div>
          <h3 className="text-gray-300 text-sm font-medium mb-1"><i className="fas fa-temperature-high mr-2"></i>实时天气</h3>
          <div className="text-5xl font-bold tracking-tighter">{weather.temperature}</div>
          <div className="text-lg text-blue-200 mt-1">{weather.condition}</div>
        </div>
        <div className="text-6xl text-yellow-400 opacity-90">
            <i className={`fas ${getIcon(weather.condition)}`}></i>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 mt-6 pt-4 border-t border-white/10">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-full bg-white/10 flex items-center justify-center text-blue-300">
            <i className="fas fa-wind"></i>
          </div>
          <div>
            <div className="text-xs text-gray-400">风速</div>
            <div className="font-semibold">{weather.windSpeed}</div>
          </div>
        </div>
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-full bg-white/10 flex items-center justify-center text-green-300">
            <i className="fas fa-compass"></i>
          </div>
          <div>
            <div className="text-xs text-gray-400">风向</div>
            <div className="font-semibold">{weather.windDirection}</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WeatherCard;