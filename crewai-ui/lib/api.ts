// lib/api.ts
export const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8000";

export interface ResearchRequest {
  query_type: string;
  product: string;
  models?: string;
  video_url?: string;
  image_url?: string;
  new_color?: string;
  topic: string;
}

export interface ApiResponse {
  status: string;
  data: {
    status?: string;
    result?: string;
    files?: string[];           // Might come from Variant Generation
    files_generated?: string[]; // Might come from Market Research
  };
  download_base_url?: string;
}

export const runFlow = async (payload: ResearchRequest): Promise<ApiResponse> => {
  const response = await fetch(`${BACKEND_URL}/api/run-flow`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Failed to run flow");
  }

  return response.json();
};

export const getFileUrl = (filename: string) => {
  // filename might be absolute path from backend, we just need the basename
  const name = filename.split(/[\\/]/).pop();
  return `${BACKEND_URL}/outputs/${name}`;
};

// For input image file upload

export const uploadFile = async (file: File): Promise<string> => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${BACKEND_URL}/api/upload`, {
    method: "POST",
    body: formData, // Notice we don't set Content-Type; the browser handles it for FormData
  });

  if (!response.ok) {
    throw new Error("Failed to upload image");
  }

  const data = await response.json();
  if (data.error) throw new Error(data.error);
  
  return data.file_path; // Returns the absolute path from the server
};