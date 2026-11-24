const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

export type PredictionMode = 'without_reviews' | 'with_reviews'

export interface ListingFeatures {
  neighbourhood_group: string
  neighbourhood: string
  latitude: number
  longitude: number
  price: number
  minimum_nights: number
  availability_365: number
  calculated_host_listings_count: number
  number_of_reviews?: number | null
  reviews_per_month?: number | null
}

export interface PredictionResponse {
  predicted_room_type: string
  predicted_class_probability: number
  probabilities: Record<string, number>
  mode: PredictionMode
  model_version: string
  disclaimer: string
}

export async function predictRoomType(mode: PredictionMode, listing: ListingFeatures): Promise<PredictionResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode, listing }),
  })

  if (!response.ok) {
    const payload = await response.json().catch(() => null)
    const detail = payload?.detail
    const message = typeof detail === 'string' ? detail : 'The prediction request could not be completed.'
    throw new Error(message)
  }

  return response.json() as Promise<PredictionResponse>
}
