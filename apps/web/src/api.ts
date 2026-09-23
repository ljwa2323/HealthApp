const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export async function uploadHealthRecord(file: File) {
  const body = new FormData();
  body.append("file", file);

  const response = await fetch(
    `${API_BASE}/api/patients/p_demo_001/documents`,
    {
      method: "POST",
      body,
    },
  );

  if (!response.ok) {
    throw new Error(`Upload failed: ${response.status}`);
  }

  return response.json();
}
