import React from 'react';
import { Attraction } from '../types';

interface AttractionListProps {
  attractions: Attraction[];
}

const AttractionList: React.FC<AttractionListProps> = ({ attractions }) => {
  if (!attractions || attractions.length === 0) return null;

  return (
    <div className="space-y-4">
      <h3 className="text-xl font-bold flex items-center">
        <i className="fas fa-map-marked-alt mr-2 text-pink-400"></i>
        附近景点
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {attractions.map((place, index) => (
          <div key={index} className="glass-panel p-4 rounded-xl hover:bg-white/10 transition-colors duration-300 border border-white/5">
            <div className="flex justify-between items-start mb-2">
              <span className="bg-pink-500/20 text-pink-200 text-xs px-2 py-1 rounded-full border border-pink-500/30">
                {place.type}
              </span>
              <div className="flex items-center space-x-2 text-sm text-gray-300">
                 <span><i className="fas fa-location-arrow transform -rotate-45"></i> {place.bearing}</span>
                 <span className="w-1 h-1 bg-gray-500 rounded-full"></span>
                 <span>{place.distance}</span>
              </div>
            </div>
            
            <h4 className="text-lg font-bold mb-1">{place.name}</h4>
            <p className="text-sm text-gray-400 line-clamp-2">{place.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AttractionList;