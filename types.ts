export interface WeatherInfo {
  temperature: string;
  condition: string;
  windSpeed: string;
  windDirection: string;
}

export interface Attraction {
  name: string;
  description: string;
  bearing: string; // e.g., "North", "NE"
  distance: string; // e.g., "500m"
  type: string;
}

export interface LocationData {
  address: string;
  locationName: string;
  weather: WeatherInfo;
  attractions: Attraction[];
}

export interface GroundingSource {
  title: string;
  uri: string;
}

export interface Coordinates {
  latitude: number;
  longitude: number;
}
