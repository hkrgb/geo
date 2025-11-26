import React, { useState, useEffect, useCallback } from 'react';
import { GeminiResponse, fetchLocationInsights } from './services/geminiService';
import { Coordinates, LocationData } from './types';
import WeatherCard from './components/WeatherCard';
import AttractionList from './components/AttractionList';
import LoadingSpinner from './components/LoadingSpinner';

const App: React.FC = () => {
  const [coords, setCoords] = useState<Coordinates | null>(null);
  const [data, setData] = useState<LocationData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [sources, setSources] = useState<{title: string, uri: string}[]>([]);

  const getLocation = useCallback(() => {
    if (!navigator.geolocation) {
      setError("您的浏览器不支持地理定位功能");
      return;
    }

    setLoading(true);
    setError(null);
    setData(null);

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const newCoords = {
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
        };
        setCoords(newCoords);
        
        // Fetch data from Gemini once we have coordinates
        fetchLocationInsights(newCoords)
          .then((response: GeminiResponse) => {
            if (response.data) {
              setData(response.data);
              setSources(response.sources);
            } else {
              setError("无法解析地理资讯，请重试");
              console.warn("Raw response:", response.rawText);
            }
          })
          .catch((err) => {
            setError("无法连接到 AI 服务: " + (err instanceof Error ? err.message : String(err)));
          })
          .finally(() => {
            setLoading(false);
          });
      },
      (err) => {
        setLoading(false);
        setError("无法获取位置权限: " + err.message);
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 0
      }
    );
  }, []);

  // Initial load
  useEffect(() => {
    // Only fetch on button click or strict user interaction is usually better for UX/Battery, 
    // but user asked for "show me where I am", so we try on load.
    getLocation();
  }, [getLocation]);

  return (
    <div className="max-w-md mx-auto min-h-screen pb-10 relative">
      {/* Background decoration */}
      <div className="fixed top-0 left-0 w-full h-full overflow-hidden -z-10 pointer-events-none">
        <div className="absolute top-0 left-1/4 w-64 h-64 bg-purple-600 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob"></div>
        <div className="absolute top-0 right-1/4 w-64 h-64 bg-yellow-600 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-2000"></div>
        <div className="absolute -bottom-8 left-1/2 w-64 h-64 bg-pink-600 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-4000"></div>
      </div>

      <header className="p-6 pb-2">
        <div className="flex justify-between items-center">
          <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-200 to-pink-200">
            <i className="fas fa-location-dot mr-2 text-white"></i>
            GeoGuide AI
          </h1>
          <button 
            onClick={getLocation} 
            disabled={loading}
            className="w-10 h-10 rounded-full glass-panel flex items-center justify-center hover:bg-white/20 transition active:scale-95 disabled:opacity-50"
          >
            <i className={`fas fa-sync-alt ${loading ? 'animate-spin' : ''}`}></i>
          </button>
        </div>
      </header>

      <main className="p-6 pt-2">
        {error && (
          <div className="bg-red-500/20 border border-red-500/50 p-4 rounded-xl text-red-100 mb-6 text-sm">
            <i className="fas fa-exclamation-circle mr-2"></i>
            {error}
            <button className="underline ml-2" onClick={getLocation}>重试</button>
          </div>
        )}

        {loading && !data && <LoadingSpinner />}

        {data && (
          <div className="animate-fade-in-up">
            {/* Location Header */}
            <div className="mb-6 text-center">
              <div className="inline-block px-3 py-1 rounded-full bg-blue-500/20 border border-blue-400/30 text-blue-200 text-xs mb-2">
                <i className="fas fa-satellite mr-1"></i>
                {coords?.latitude.toFixed(4)}, {coords?.longitude.toFixed(4)}
              </div>
              <h2 className="text-3xl font-bold text-white leading-tight mb-1 drop-shadow-lg">
                {data.locationName}
              </h2>
              <p className="text-gray-400 text-sm">{data.address}</p>
            </div>

            {/* Weather Widget */}
            <WeatherCard weather={data.weather} />

            {/* Attractions */}
            <AttractionList attractions={data.attractions} />
            
            {/* Sources / Attribution */}
            {sources.length > 0 && (
              <div className="mt-8 pt-6 border-t border-white/10">
                <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">资料来源</h4>
                <ul className="space-y-1">
                  {sources.map((source, idx) => (
                    <li key={idx}>
                      <a 
                        href={source.uri} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="text-xs text-blue-400 hover:text-blue-300 truncate block transition-colors"
                      >
                        <i className="fas fa-external-link-alt mr-1 text-[10px]"></i>
                        {source.title}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {!loading && !data && !error && (
          <div className="text-center text-gray-400 py-20">
            <div className="text-6xl mb-4 opacity-30"><i className="fas fa-map-marked"></i></div>
            <p>点击右上角刷新按钮获取您的当前位置资讯</p>
          </div>
        )}
      </main>
    </div>
  );
};

export default App;