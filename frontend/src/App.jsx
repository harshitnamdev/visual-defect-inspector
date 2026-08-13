import { useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const CATEGORIES = ['screw', 'bottle', 'leather', 'hazelnut']

export default function App() {
  const [category, setCategory] = useState('screw')
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  function handleFileChange(e) {
    const selected = e.target.files[0]
    if (!selected) return
    setFile(selected)
    setPreviewUrl(URL.createObjectURL(selected))
    setResult(null)
    setError(null)
  }

  async function handleSubmit() {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('category', category)
    formData.append('image', file)

    try {
      const res = await fetch(`${API_URL}/detect`, { method: 'POST', body: formData })
      if (!res.ok) throw new Error('Detection request failed')
      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError('Something went wrong. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        <h1 style={styles.title}>Visual Defect Inspector</h1>
        <p style={styles.subtitle}>
          Upload a photo of a product to check it for visual defects, with the flagged region highlighted.
        </p>

        <div style={styles.controls}>
          <label style={styles.label}>
            Category
            <select value={category} onChange={(e) => setCategory(e.target.value)} style={styles.select}>
              {CATEGORIES.map((c) => (
                <option key={c} value={c}>
                  {c.charAt(0).toUpperCase() + c.slice(1)}
                </option>
              ))}
            </select>
          </label>

          <label style={styles.label}>
            Photo
            <input type="file" accept="image/*" onChange={handleFileChange} style={styles.fileInput} />
          </label>

          <button onClick={handleSubmit} disabled={!file || loading} style={styles.button}>
            {loading ? 'Inspecting...' : 'Inspect'}
          </button>
        </div>

        {error && <div style={styles.error}>{error}</div>}

        {previewUrl && (
          <div style={styles.imagesRow}>
            <div style={styles.imageBlock}>
              <div style={styles.imageCaption}>Original</div>
              <img src={previewUrl} alt="original upload" style={styles.image} />
            </div>
            {result && (
              <div style={styles.imageBlock}>
                <div style={styles.imageCaption}>Detection result</div>
                <img
                  src={`data:image/png;base64,${result.overlay_image_base64}`}
                  alt="detection heatmap overlay"
                  style={styles.image}
                />
              </div>
            )}
          </div>
        )}

        {result && (
          <div style={{ ...styles.resultCard, borderColor: result.label === 'Defective' ? '#c0392b' : '#2e8b57' }}>
            <div style={{ ...styles.resultLabel, color: result.label === 'Defective' ? '#c0392b' : '#2e8b57' }}>
              {result.label}
            </div>
            <div style={styles.resultMeta}>
              anomaly score {result.score} (threshold {result.threshold}) &middot; region: {result.region}
            </div>
            <div style={styles.narrative}>{result.narrative}</div>
          </div>
        )}
      </div>
    </div>
  )
}

const styles = {
  page: { minHeight: '100vh', background: '#f4f6fb', fontFamily: "'Segoe UI', Arial, sans-serif", padding: '32px 16px' },
  container: { maxWidth: 760, margin: '0 auto' },
  title: { fontSize: 28, color: '#0b3d91', marginBottom: 4 },
  subtitle: { color: '#555', marginBottom: 24 },
  controls: { display: 'flex', gap: 16, alignItems: 'flex-end', flexWrap: 'wrap', marginBottom: 20 },
  label: { display: 'flex', flexDirection: 'column', fontSize: 13, color: '#333', gap: 4 },
  select: { padding: '8px 10px', borderRadius: 6, border: '1px solid #ccd4e6', fontSize: 14 },
  fileInput: { fontSize: 13 },
  button: {
    padding: '10px 20px', borderRadius: 6, border: 'none', background: '#0b3d91',
    color: 'white', fontSize: 14, cursor: 'pointer', height: 38,
  },
  error: { color: '#c0392b', marginBottom: 16 },
  imagesRow: { display: 'flex', gap: 20, flexWrap: 'wrap', marginBottom: 20 },
  imageBlock: { flex: '1 1 300px' },
  imageCaption: { fontSize: 12, color: '#666', marginBottom: 6, textTransform: 'uppercase', letterSpacing: 0.5 },
  image: { width: '100%', borderRadius: 8, border: '1px solid #dbe2f0' },
  resultCard: { background: 'white', border: '2px solid', borderRadius: 8, padding: 16 },
  resultLabel: { fontSize: 20, fontWeight: 600, marginBottom: 4 },
  resultMeta: { fontSize: 12, color: '#777', marginBottom: 10 },
  narrative: { fontSize: 14, color: '#222' },
}
