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
      我现在的位置是 Latitude: ${coords.latitude}, Longitude: ${coords.longitude}.
      
      请利用 Google Search 和 Google Maps 帮我完成以下任务:
      1. 确认我现在所在的具体地名和街道 (Address & Location Name).
      2. 搜寻此位置现在的实时天气 (Temperature, Condition, Wind Speed, Wind Direction).
      3. 搜寻附近 4-5 个值得去的景点或地标.
      4. 对于每个景点，请计算或估计它相对于我现在的方位的方向 (例如: 东北, 南, 西北) 和大约距离.

      请以纯 JSON 格式输出结果，格式如下 (不要包含任何 markdown 格式以外的文字):
      \`\`\`json
      {
        "locationName": "当前地名",
        "address": "当前完整地址",
        "weather": {
          "temperature": "25°C",
          "condition": "多云",
          "windSpeed": "15 km/h",
          "windDirection": "西北"
        },
        "attractions": [
          {
            "name": "景点名称",
            "description": "简短描述 (20字以内)",
            "bearing": "方向 (例如: 东北)",
            "distance": "距离 (例如: 500m)",
            "type": "类别 (例如: 公园, 餐厅, 博物馆)"
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
