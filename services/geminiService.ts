import { GoogleGenAI } from "@google/genai";
import { LocationData, Coordinates, GroundingSource } from "../types";

const apiKey = process.env.API_KEY || '';
const ai = new GoogleGenAI({ apiKey });

export interface GeminiResponse {
  data: LocationData | null;
  sources: GroundingSource[];
  rawText: string;
}

export const fetchLocationInsights = async (coords: Coordinates): Promise<GeminiResponse> => {
  try {
    const modelId = "gemini-2.5-flash";
    
    // We cannot force JSON schema when using tools (Google Search/Maps), 
    // so we prompt strongly for a JSON code block and parse it manually.
    const prompt = `
      我現在的位置是 Latitude: ${coords.latitude}, Longitude: ${coords.longitude}.
      
      請利用 Google Search 和 Google Maps 幫我完成以下任務 (請使用繁體中文/廣東話回答):
      1. 確認我現在所在的具體地名和街道 (Address & Location Name).
      2. 搜尋此位置現在的實時天氣 (Temperature, Condition, Wind Speed, Wind Direction).
      3. 搜尋附近 4-5 個值得去的景點或地標.
      4. 對於每個景點，請計算或估計它相對於我現在的方位的方向 (例如: 東北, 南, 西北) 和大約距離.

      請以純 JSON 格式輸出結果，格式如下 (不要包含任何 markdown 格式以外的文字):
      \`\`\`json
      {
        "locationName": "當前地名",
        "address": "當前完整地址",
        "weather": {
          "temperature": "25°C",
          "condition": "多雲",
          "windSpeed": "15 km/h",
          "windDirection": "西北"
        },
        "attractions": [
          {
            "name": "景點名稱",
            "description": "簡短描述 (20字以內)",
            "bearing": "方向 (例如: 東北)",
            "distance": "距離 (例如: 500m)",
            "type": "類別 (例如: 公園, 餐廳, 博物館)"
          }
        ]
      }
      \`\`\`
    `;

    const response = await ai.models.generateContent({
      model: modelId,
      contents: prompt,
      config: {
        tools: [{ googleSearch: {} }, { googleMaps: {} }],
        toolConfig: {
          retrievalConfig: {
            latLng: {
              latitude: coords.latitude,
              longitude: coords.longitude
            }
          }
        }
        // Note: responseSchema and responseMimeType are NOT allowed when using tools
      },
    });

    const text = response.text || "";
    
    // Extract sources from grounding chunks
    const sources: GroundingSource[] = [];
    const chunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks;
    
    if (chunks) {
      chunks.forEach((chunk: any) => {
        if (chunk.web?.uri && chunk.web?.title) {
          sources.push({ title: chunk.web.title, uri: chunk.web.uri });
        }
        if (chunk.maps?.uri && chunk.maps?.title) {
           sources.push({ title: chunk.maps.title, uri: chunk.maps.uri });
        }
      });
    }

    // Parse JSON from the text response
    let parsedData: LocationData | null = null;
    try {
      // Find JSON block
      const jsonMatch = text.match(/```json\s*([\s\S]*?)\s*```/) || text.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const jsonStr = jsonMatch[1] || jsonMatch[0];
        parsedData = JSON.parse(jsonStr);
      }
    } catch (e) {
      console.error("Failed to parse JSON from Gemini response:", e);
      // Fallback or partial error handling could go here
    }

    return {
      data: parsedData,
      sources: sources,
      rawText: text // Useful for debugging or fallback display
    };

  } catch (error) {
    console.error("Gemini API Error:", error);
    throw error;
  }
};