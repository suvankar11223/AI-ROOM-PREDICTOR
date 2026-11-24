import { useEffect, useMemo, useState, type FormEvent } from 'react'
import {
  ArrowRight,
  BedDouble,
  Building2,
  ChevronDown,
  Info,
  MapPin,
  Sparkles,
  Waves
} from 'lucide-react'
import neighbourhoods from './data/neighbourhoods.json'
import metrics from './data/model-metrics.json'

type Mode = 'without_reviews' | 'with_reviews'
type Room = 'Entire home/apt' | 'Private room' | 'Shared room'
type Page = 'Overview' | 'Predictor' | 'Model Insights'
type ListingFormData = {
  neighbourhood_group: string
  neighbourhood: string
  latitude: string
  longitude: string
  price: string
  minimum_nights: string
  availability_365: string
  calculated_host_listings_count: string
  number_of_reviews: string
  reviews_per_month: string
}
type Result = {
  predicted_room_type: Room
  predicted_class_probability: number
  probabilities: Record<Room, number>
  mode: Mode
  model_version: string
  disclaimer: string
}
type InputProps = {
  label: string
  value: string
  onChange: (value: string) => void
  min?: number
  max?: number
  step?: string
  required?: boolean
  unit?: string
}

const classes: Room[] = ['Entire home/apt', 'Private room', 'Shared room']
const roomTypeClass: Record<Room, string> = {
  'Entire home/apt': 'entire',
  'Private room': 'private',
  'Shared room': 'shared'
}
const boroughs = Object.keys(neighbourhoods) as Array<keyof typeof neighbourhoods>
const initial: ListingFormData = {
  neighbourhood_group: 'Manhattan',
  neighbourhood: 'Midtown',
  latitude: '40.7549',
  longitude: '-73.9840',
  price: '180',
  minimum_nights: '2',
  availability_365: '120',
  calculated_host_listings_count: '1',
  number_of_reviews: '48',
  reviews_per_month: '2.4'
}
const API = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
const pct = (value: number) => `${(value * 100).toFixed(1)}%`

function Input({
  label,
  value,
  onChange,
  min,
  max,
  step = 'any',
  required = true,
  unit
}: InputProps) {
  return (
    <label className="field">
      <span className="field-label">
        {label}
        {unit && <span className="unit">{unit}</span>}
      </span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        required={required}
        onChange={event => onChange(event.target.value)}
      />
    </label>
  )
}

function Skyline() {
  const buildings = [
    [20, 115, 40, 80], [68, 98, 38, 97], [113, 126, 34, 69],
    [155, 83, 46, 112], [210, 103, 34, 92], [252, 58, 50, 137],
    [311, 92, 38, 103], [358, 70, 47, 125], [414, 105, 34, 90],
    [458, 78, 47, 117], [515, 96, 36, 99], [560, 62, 50, 133],
    [619, 103, 37, 92], [665, 80, 45, 115], [720, 110, 34, 85]
  ]

  return (
    <svg viewBox="0 0 820 220" role="img" aria-label="New York City skyline illustration">
      {buildings.map(([x, y, width, height]) => (
        <g key={x}>
          <rect x={x} y={y} width={width} height={height} fill="#efece3" stroke="#c9c2b1" />
          {Array.from({ length: Math.floor(height / 15) }, (_, row) =>
            Array.from({ length: Math.floor(width / 10) }, (_, column) => (
              <rect
                key={`${row}-${column}`}
                x={x + 5 + column * 10}
                y={y + 8 + row * 15}
                width="3"
                height="5"
                fill="#c0430d"
                opacity=".45"
              />
            ))
          )}
        </g>
      ))}
      <path d="M252 60V0m-7 60h28m-17-60h8m-4 0v-9M568 65V4m-6 61h20" stroke="#8a6212" strokeWidth="2" />
      <path d="M0 198H820" stroke="#c9c2b1" strokeWidth="2" />
    </svg>
  )
}

function Buildings({ active, loading }: { active: Room | null; loading: boolean }) {
  const buildings = [
    { room: classes[0], x: 48, width: 82, height: 157 },
    { room: classes[1], x: 175, width: 91, height: 137 },
    { room: classes[2], x: 312, width: 80, height: 108 }
  ]

  return (
    <svg
      className={`building-art${loading ? ' pulse' : ''}`}
      viewBox="0 0 440 205"
      role="img"
      aria-label="Room type comparison"
    >
      {buildings.map(({ room, x, width, height }) => {
        const y = 179 - height
        const selected = active === room
        const color = roomTypeClass[room]

        return (
          <g key={room} className={selected ? `chosen ${color}` : undefined}>
            <path
              d={`M${x + width / 2} ${y}l-48 -${selected ? 72 : 0}h96z`}
              fill={selected ? '#c0430d' : 'transparent'}
              opacity={selected ? 0.18 : 0}
            />
            <rect
              x={x - 7}
              y={y - 6}
              width={width + 14}
              height={height + 8}
              rx="3"
              fill={selected ? '#c0430d' : 'transparent'}
              opacity={selected ? 0.1 : 0}
            />
            <rect
              x={x}
              y={y}
              width={width}
              height={height}
              rx="2"
              fill="#f4f1ea"
              stroke={selected ? '#c0430d' : '#9a9386'}
              strokeWidth={selected ? 2 : 1}
            />
            <path
              d={`M${x - 5} ${y}h${width + 10}l-7 -9h${-width + 4}`}
              fill="#e7e2d6"
              stroke="#9a9386"
            />
            {Array.from({ length: Math.floor(height / 32) }, (_, row) =>
              Array.from({ length: 3 }, (_, column) => (
                <rect
                  key={`${row}-${column}`}
                  x={x + 12 + column * ((width - 27) / 3)}
                  y={y + 13 + row * 31}
                  width="13"
                  height="18"
                  rx="2"
                  fill={selected ? '#c0430d' : '#8a6212'}
                  opacity={selected ? 0.85 : 0.55}
                />
              ))
            )}
          </g>
        )
      })}
      <path d="M20 180H420" stroke="#9a9386" />
    </svg>
  )
}

function Hero({
  page,
  onPredict
}: {
  page: Page
  onPredict?: () => void
}) {
  const isPredictor = page === 'Predictor'

  return (
    <section className={`hero${isPredictor ? '' : ' hero-page'}`}>
      <div className="hero-copy">
        <p className="kicker">
          {page === 'Overview'
            ? 'STAYWISE / NYC LISTING RESEARCH'
            : page === 'Model Insights'
              ? `MODEL TRANSPARENCY / VERSION ${metrics.model_version}`
              : 'NEW YORK CITY / LISTING CLASSIFICATION'}
        </p>
        <h1>
          {page === 'Model Insights' ? (
            <>Look beyond the <em>prediction.</em></>
          ) : (
            <>Every listing tells a <em>different story.</em></>
          )}
        </h1>
        <p className="lede">
          {isPredictor
            ? 'Estimate the room type of an NYC Airbnb listing using its listing attributes.'
            : page === 'Overview'
              ? 'Explore how listing attributes relate to room-type patterns across New York City.'
              : 'Review host-grouped holdout results from the 2019 NYC Airbnb snapshot. These are research estimates, not production guarantees.'}
        </p>
        {isPredictor ? (
          <div className="hero-meta">
            <span>2019 NYC snapshot</span>
            <span>3 room types</span>
            <span>2 model modes</span>
          </div>
        ) : (
          onPredict && (
            <div className="hero-actions">
              <button className="btn btn-primary" onClick={onPredict}>
                Open predictor <ArrowRight size={16} />
              </button>
            </div>
          )
        )}
      </div>
      {isPredictor && (
        <figure className="hero-figure">
          <Skyline />
          <figcaption>
            <span>NEW YORK CITY</span>
            <span>2019 AIRBNB SNAPSHOT</span>
          </figcaption>
        </figure>
      )}
    </section>
  )
}

export default function App() {
  const [page, setPage] = useState<Page>('Predictor')
  const [mode, setMode] = useState<Mode>('without_reviews')
  const [form, setForm] = useState<ListingFormData>(initial)
  const [result, setResult] = useState<Result | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [api, setApi] = useState<'checking' | 'online' | 'offline'>('checking')

  const options = useMemo(
    () => neighbourhoods[form.neighbourhood_group as keyof typeof neighbourhoods] || [],
    [form.neighbourhood_group]
  )

  useEffect(() => {
    void fetch(`${API}/api/v1/health`)
      .then(response => (response.ok ? response.json() : Promise.reject()))
      .then(health => setApi(health.models_loaded ? 'online' : 'offline'))
      .catch(() => setApi('offline'))
  }, [])

  function updateField<K extends keyof ListingFormData>(key: K, value: ListingFormData[K]) {
    setForm(current => ({ ...current, [key]: value }))
    setResult(null)
  }

  async function predict(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError('')
    setResult(null)
    setLoading(true)
    const start = Date.now()

    try {
      const listing: Record<string, string | number | null> = {
        neighbourhood_group: form.neighbourhood_group,
        neighbourhood: form.neighbourhood,
        latitude: Number(form.latitude),
        longitude: Number(form.longitude),
        price: Number(form.price),
        minimum_nights: Number(form.minimum_nights),
        availability_365: Number(form.availability_365),
        calculated_host_listings_count: Number(form.calculated_host_listings_count)
      }

      if (mode === 'with_reviews') {
        listing.number_of_reviews = Number(form.number_of_reviews)
        listing.reviews_per_month =
          form.reviews_per_month.trim() === '' ? null : Number(form.reviews_per_month)
      }

      const response = await fetch(`${API}/api/v1/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode, listing })
      })
      const body = await response.json()

      if (!response.ok) {
        throw new Error(
          typeof body.detail === 'string'
            ? body.detail
            : 'Prediction request rejected. Check the input values.'
        )
      }

      await new Promise(resolve =>
        window.setTimeout(resolve, Math.max(0, 2000 - (Date.now() - start)))
      )
      setResult(body as Result)
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : 'Could not connect to the prediction API.'
      )
    } finally {
      await new Promise(resolve =>
        window.setTimeout(resolve, Math.max(0, 2000 - (Date.now() - start)))
      )
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar-inner">
          <button className="brand" onClick={() => setPage('Overview')} aria-label="Staywise home">
            <Building2 className="brand-mark" size={19} />
            <span className="brand-name">Staywise<span className="dot">.</span></span>
          </button>
          <span className="brand-divider" />
          <small className="brand-caption">NYC AIRBNB ROOM TYPE PREDICTOR</small>
          <nav aria-label="Main navigation">
            {(['Overview', 'Predictor', 'Model Insights'] as const).map(item => (
              <button
                key={item}
                className={page === item ? 'active' : ''}
                onClick={() => setPage(item)}
                aria-current={page === item ? 'page' : undefined}
              >
                {item}
              </button>
            ))}
          </nav>
          <div className="api-state" aria-live="polite">
            <span className={`dot ${api}`} />
            <span className="api-label">
              {api === 'online'
                ? 'API connected'
                : api === 'checking'
                  ? 'Checking API'
                  : 'API unavailable'}
            </span>
          </div>
        </div>
      </header>

      {page === 'Predictor' && (
        <main className="page">
          <Hero page={page} />
          <section className="workspace">
            <form className="panel form-panel" onSubmit={predict}>
              <div className="panel-head">
                <div>
                  <h2>Listing details</h2>
                  <p>Provide the listing information for prediction.</p>
                </div>
                <button
                  type="button"
                  className="btn btn-ghost"
                  onClick={() => {
                    setForm(initial)
                    setResult(null)
                    setError('')
                  }}
                >
                  Use example
                </button>
              </div>

              <div className="segmented" role="group" aria-label="Prediction mode">
                <button
                  type="button"
                  className={mode === 'without_reviews' ? 'on' : ''}
                  onClick={() => {
                    setMode('without_reviews')
                    setResult(null)
                  }}
                  aria-pressed={mode === 'without_reviews'}
                >
                  Without reviews <span className="seg-meta">8 features</span>
                </button>
                <button
                  type="button"
                  className={mode === 'with_reviews' ? 'on' : ''}
                  onClick={() => {
                    setMode('with_reviews')
                    setResult(null)
                  }}
                  aria-pressed={mode === 'with_reviews'}
                >
                  With reviews <span className="seg-meta">10 features</span>
                </button>
              </div>

              <section className="form-section">
                <h3><MapPin size={16} /> Location</h3>
                <div className="fields">
                  <label className="field">
                    <span className="field-label">Borough</span>
                    <div className="select">
                      <select
                        value={form.neighbourhood_group}
                        onChange={event => {
                          const borough = event.target.value as keyof typeof neighbourhoods
                          setForm(current => ({
                            ...current,
                            neighbourhood_group: borough,
                            neighbourhood: neighbourhoods[borough][0]
                          }))
                          setResult(null)
                        }}
                      >
                        {boroughs.map(borough => <option key={borough}>{borough}</option>)}
                      </select>
                      <ChevronDown size={16} />
                    </div>
                  </label>
                  <label className="field">
                    <span className="field-label">Neighbourhood</span>
                    <div className="select">
                      <select
                        value={form.neighbourhood}
                        onChange={event => updateField('neighbourhood', event.target.value)}
                      >
                        {options.map(neighbourhood => (
                          <option key={neighbourhood}>{neighbourhood}</option>
                        ))}
                      </select>
                      <ChevronDown size={16} />
                    </div>
                  </label>
                  <Input
                    label="Latitude"
                    value={form.latitude}
                    onChange={value => updateField('latitude', value)}
                    min={-90}
                    max={90}
                    step="0.0001"
                  />
                  <Input
                    label="Longitude"
                    value={form.longitude}
                    onChange={value => updateField('longitude', value)}
                    min={-180}
                    max={180}
                    step="0.0001"
                  />
                </div>
              </section>

              <section className="form-section">
                <h3><BedDouble size={16} /> Listing attributes</h3>
                <div className="fields">
                  <Input
                    label="Nightly price"
                    value={form.price}
                    onChange={value => updateField('price', value)}
                    min={0}
                    unit="USD"
                  />
                  <Input
                    label="Minimum nights"
                    value={form.minimum_nights}
                    onChange={value => updateField('minimum_nights', value)}
                    min={0}
                    step="1"
                  />
                  <Input
                    label="Availability"
                    value={form.availability_365}
                    onChange={value => updateField('availability_365', value)}
                    min={0}
                    max={365}
                    step="1"
                    unit="days/year"
                  />
                  <Input
                    label="Host listings count"
                    value={form.calculated_host_listings_count}
                    onChange={value => updateField('calculated_host_listings_count', value)}
                    min={0}
                    step="1"
                  />
                </div>
              </section>

              {mode === 'with_reviews' && (
                <section className="form-section review" aria-label="Review history">
                  <h3><Waves size={16} /> Review history</h3>
                  <div className="fields">
                    <Input
                      label="Number of reviews"
                      value={form.number_of_reviews}
                      onChange={value => updateField('number_of_reviews', value)}
                      min={0}
                      step="1"
                    />
                    <Input
                      label="Reviews per month"
                      value={form.reviews_per_month}
                      onChange={value => updateField('reviews_per_month', value)}
                      min={0}
                      step="0.1"
                      required={form.number_of_reviews !== '0'}
                    />
                  </div>
                  <small className="hint">For zero reviews, leave reviews per month blank.</small>
                </section>
              )}

              {error && <div className="error" role="alert">{error}</div>}
              <button className="btn btn-primary btn-block" disabled={loading}>
                {loading ? (
                  <>
                    <span className="spinner" />
                    Predicting...
                  </>
                ) : (
                  <>
                    Predict room type <Sparkles size={17} /> <ArrowRight size={16} />
                  </>
                )}
              </button>
              <div className="api-note" aria-live="polite">
                <span className={`dot ${api}`} />
                {api === 'online'
                  ? 'Prediction API connected'
                  : api === 'checking'
                    ? 'Checking prediction API...'
                    : 'API unavailable - start FastAPI on port 8000'}
              </div>
            </form>

            <aside className="panel result" aria-label="Prediction results">
              <div className="building-wrap">
                <Buildings active={result?.predicted_room_type ?? null} loading={loading} />
                <div className="building-labels">
                  {classes.map(room => (
                    <span
                      key={room}
                      className={`${roomTypeClass[room]}${result?.predicted_room_type === room ? ' selected' : ''}`}
                    >
                      {room}
                    </span>
                  ))}
                </div>
              </div>

              {result ? (
                <div className="readout">
                  <p className="readout-label">Predicted room type</p>
                  <p className={`readout-value c-${roomTypeClass[result.predicted_room_type]}`}>
                    {result.predicted_room_type}
                  </p>
                  <p className="readout-note">
                    {pct(result.predicted_class_probability)} predicted probability
                    <span className="mono"> / model {result.model_version}</span>
                  </p>
                </div>
              ) : (
                <div className="readout readout--empty">
                  <span className="glyph"><Sparkles size={18} /></span>
                  <div>
                    <h3>{loading ? 'Reading the listing...' : 'No prediction yet'}</h3>
                    <p>
                      {loading
                        ? 'The model is evaluating the supplied listing attributes.'
                        : 'Fill in the listing details and predict to see the model output here.'}
                    </p>
                  </div>
                </div>
              )}

              <section className="probabilities">
                <h3>Class probabilities</h3>
                {classes.map(room => (
                  <div
                    key={room}
                    className={`prob-row ${roomTypeClass[room]}${result?.predicted_room_type === room ? ' is-predicted' : ''}`}
                  >
                    <span className="prob-name">{room}</span>
                    <div className="prob-track">
                      <i style={{ width: result ? pct(result.probabilities[room]) : '0%' }} />
                    </div>
                    <b className="prob-val">
                      {result ? pct(result.probabilities[room]) : '-'}
                    </b>
                  </div>
                ))}
              </section>

              <p className="disclaimer">
                <Info size={15} />
                Results are research estimates based on the 2019 NYC Airbnb dataset.
              </p>
            </aside>
          </section>
        </main>
      )}

      {page === 'Overview' && (
        <main className="page">
          <Hero page={page} onPredict={() => setPage('Predictor')} />
          <section className="ledger" aria-label="Dataset summary">
            <div className="ledger-cell">
              <strong>48,895</strong><span>NYC listings</span><small>2019 snapshot</small>
            </div>
            <div className="ledger-cell">
              <strong>5</strong><span>Boroughs</span><small>NYC geography</small>
            </div>
            <div className="ledger-cell">
              <strong>3</strong><span>Room types</span><small>Classification target</small>
            </div>
            <div className="ledger-cell">
              <strong>2</strong><span>Prediction modes</span><small>With / without reviews</small>
            </div>
          </section>
          <section className="split">
            <div>
              <h2>How to use the predictor</h2>
              <p>Enter the listing details, choose a feature set, and review the model's estimated room type.</p>
              <ol className="steps">
                <li><span className="step-no">01</span><div><h3>Describe the listing</h3><p>Set the location and listing attributes using the form.</p></div></li>
                <li><span className="step-no">02</span><div><h3>Choose a model</h3><p>Use review history when it's available, or choose the cold-start model.</p></div></li>
                <li><span className="step-no">03</span><div><h3>Review estimates</h3><p>Compare the predicted class with the probabilities for all three room types.</p></div></li>
              </ol>
            </div>
            <div>
              <h2>Model inputs</h2>
              <p>The prediction uses listing attributes from the NYC Airbnb 2019 snapshot.</p>
              <ul className="feature-list">
                <li>Location <span>Borough, neighbourhood, coordinates</span></li>
                <li>Availability <span>Price, minimum nights, days/year</span></li>
                <li>Host <span>Host listings count</span></li>
                <li>Reviews <span>Optional review history</span></li>
              </ul>
            </div>
          </section>
          <p className="note">
            These are research estimates from a 2019 cross-sectional dataset, not guarantees of current room type.
          </p>
        </main>
      )}

      {page === 'Model Insights' && (
        <main className="page">
          <Hero page={page} />
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th scope="col">Metric</th>
                  <th scope="col">Without reviews</th>
                  <th scope="col">With reviews</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th scope="row">Accuracy</th>
                  <td>{pct(metrics.models.without_reviews.accuracy)}</td>
                  <td>{pct(metrics.models.with_reviews.accuracy)}</td>
                </tr>
                <tr>
                  <th scope="row">Macro F1</th>
                  <td>{pct(metrics.models.without_reviews.macro_f1)}</td>
                  <td>{pct(metrics.models.with_reviews.macro_f1)}</td>
                </tr>
                <tr>
                  <th scope="row">Balanced accuracy</th>
                  <td>{pct(metrics.models.without_reviews.balanced_accuracy)}</td>
                  <td>{pct(metrics.models.with_reviews.balanced_accuracy)}</td>
                </tr>
                <tr>
                  <th scope="row">Shared room recall</th>
                  <td>{pct(metrics.models.without_reviews.shared_room_recall)}</td>
                  <td>{pct(metrics.models.with_reviews.shared_room_recall)}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <section className="gates">
            <h2>Limitations and open validation gates</h2>
            <p>
              Shared-room recall is about 28.8% in both scenarios. The classifier misses a substantial share of this minority class.
            </p>
            <ol>
              {metrics.unresolved_production_gates.map(gate => <li key={gate}>{gate}</li>)}
            </ol>
          </section>
        </main>
      )}

      <footer>
        <div className="footer-inner">
          <b>Staywise<span>.</span></b>
          <small>Research estimates based on the 2019 NYC Airbnb dataset.</small>
          <span className="footer-meta">Not a guarantee of actual room type.</span>
        </div>
      </footer>
    </div>
  )
}
