import React, { useState } from 'react';
import { Upload, FileText, CheckCircle, AlertCircle, Eye, Database } from 'lucide-react';

const API_URL = "http://localhost:8000/api/v1";

export default function App() {
    const [file, setFile] = useState(null);
    const [licId, setLicId] = useState("LIC-TEST-001");
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const handleFileChange = (e) => {
        if (e.target.files) {
            setFile(e.target.files[0]);
        }
    };

    const handleUpload = async () => {
        if (!file) return;

        setLoading(true);
        setError(null);
        setResult(null);

        const formData = new FormData();
        formData.append("file", file);
        formData.append("lic_id", licId);

        try {
            const response = await fetch(`${API_URL}/licitaciones/ingest`, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Error: ${response.statusText}`);
            }

            const data = await response.json();

            // Fetch details immediately to show full data
            const detailResponse = await fetch(`${API_URL}/licitaciones/${data.licitacion_id}`);
            const detailData = await detailResponse.json();

            setResult(detailData);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-slate-950 text-slate-200 p-8 font-mono">
            <div className="max-w-6xl mx-auto space-y-8">

                {/* HEADER */}
                <header className="border-b border-slate-800 pb-4">
                    <h1 className="text-3xl font-bold text-blue-400 flex items-center gap-2">
                        <Database className="w-8 h-8" />
                        Licitaciones ETL Debugger
                    </h1>
                    <p className="text-slate-500 mt-2">PDF Ingestion & Visual Analysis Pipeline</p>
                </header>

                {/* INPUT SECTION */}
                <section className="bg-slate-900 p-6 rounded-lg border border-slate-800 space-y-4">
                    <h2 className="text-xl font-semibold flex items-center gap-2">
                        <Upload className="w-5 h-5 text-emerald-400" />
                        Ingest Pipeline
                    </h2>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                            <label className="block text-sm text-slate-400 mb-1">Licitación ID</label>
                            <input
                                type="text"
                                value={licId}
                                onChange={(e) => setLicId(e.target.value)}
                                className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-white focus:ring-2 focus:ring-blue-500 outline-none"
                            />
                        </div>
                        <div>
                            <label className="block text-sm text-slate-400 mb-1">PDF File</label>
                            <input
                                type="file"
                                accept=".pdf"
                                onChange={handleFileChange}
                                className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-300 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700"
                            />
                        </div>
                    </div>

                    <button
                        onClick={handleUpload}
                        disabled={!file || loading}
                        className={`w-full py-3 rounded-lg font-bold transition-all ${loading ? 'bg-slate-700 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-500 text-white'}`}
                    >
                        {loading ? "Processing Pipeline..." : "RUN INGESTION"}
                    </button>

                    {error && (
                        <div className="p-4 bg-red-900/30 border border-red-800 text-red-300 rounded flex items-center gap-2">
                            <AlertCircle size={20} />
                            {error}
                        </div>
                    )}
                </section>

                {/* RESULTS SECTION */}
                {result && (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

                        {/* LEFT: VISUAL METADATA */}
                        <section className="bg-slate-900 p-6 rounded-lg border border-slate-800 h-fit">
                            <h2 className="text-xl font-semibold flex items-center gap-2 mb-4 text-purple-400">
                                <Eye className="w-5 h-5" />
                                Visual Intelligence (Florence-2)
                            </h2>
                            <div className="space-y-4">
                                {result.documentos?.map((doc, idx) => {
                                    const metadata = typeof doc.metadata === 'string' ? JSON.parse(doc.metadata) : doc.metadata;
                                    const visual = metadata.visual_content || {};

                                    return (
                                        <div key={idx} className="bg-slate-950 p-4 rounded border border-slate-700">
                                            <h3 className="font-bold text-slate-300 mb-2">{doc.nombre_archivo}</h3>
                                            {Object.keys(visual).length > 0 ? (
                                                <div className="space-y-3">
                                                    {Object.entries(visual).map(([page, info]) => (
                                                        <div key={page} className="text-sm">
                                                            <span className="text-blue-400 font-bold">Page {page}:</span>
                                                            <p className="text-slate-400 mt-1 pl-4 border-l-2 border-slate-700">
                                                                {info.visual_description}
                                                            </p>
                                                        </div>
                                                    ))}
                                                </div>
                                            ) : (
                                                <p className="text-slate-600 italic">No visual metadata found.</p>
                                            )}
                                        </div>
                                    );
                                })}
                            </div>
                        </section>

                        {/* RIGHT: JSON OUTPUT */}
                        <section className="bg-slate-900 p-6 rounded-lg border border-slate-800">
                            <h2 className="text-xl font-semibold flex items-center gap-2 mb-4 text-emerald-400">
                                <FileText className="w-5 h-5" />
                                Extracted Data (JSON)
                            </h2>
                            <pre className="bg-slate-950 p-4 rounded border border-slate-700 overflow-auto max-h-[600px] text-xs text-green-300">
                                {JSON.stringify(result, null, 2)}
                            </pre>
                        </section>

                    </div>
                )}
            </div>
        </div>
    );
}
