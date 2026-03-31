package com.climate.backend.controller;


import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.Map;

/**
 * ClimateController
 * Accepts climate input from the frontend/Postman,
 * forwards it to the Python FastAPI ML service,
 * and returns the cascading impact prediction.
 */
@RestController
@RequestMapping("/api/climate")
@CrossOrigin(origins = "*") // Allow frontend/Postman to connect
public class ClimateController {

    // Python FastAPI URL — change port if needed
    private static final String PYTHON_API_URL = "http://127.0.0.1:8000/predict";

    private final RestTemplate restTemplate = new RestTemplate();


    /**
     * POST /api/climate/predict
     * Input  : { "temp": 40, "soil_moisture": 30, "ndvi": 0.3, "humidity": 60 }
     * Output : { "crop_loss": 23.4, "energy_load": 1100, "transport_delay": 15.2,
     *            "cascading_alert": true, "alert_message": "..." }
     */
    @PostMapping("/predict")
    public ResponseEntity<Map<String, Object>> getClimateImpact(
            @RequestBody ClimateRequest request) {

        try {
            // Build HTTP headers
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            // Build request body for Python
            Map<String, Object> pythonPayload = new HashMap<>();
            pythonPayload.put("temp",          request.getTemp());
            pythonPayload.put("soil_moisture", request.getSoilMoisture());
            pythonPayload.put("ndvi",          request.getNdvi());
            pythonPayload.put("humidity",      request.getHumidity());

            HttpEntity<Map<String, Object>> entity =
                    new HttpEntity<>(pythonPayload, headers);

            // Call Python FastAPI
            ResponseEntity<Map> pythonResponse =
                    restTemplate.postForEntity(PYTHON_API_URL, entity, Map.class);

            Map<String, Object> prediction = pythonResponse.getBody();

            // Add cascading impact logic on top of ML output
            if (prediction != null) {
                prediction = addCascadingAnalysis(prediction, request);
            }

            return ResponseEntity.ok(prediction);

        } catch (Exception e) {
            Map<String, Object> error = new HashMap<>();
            error.put("error", "Could not reach Python ML service.");
            error.put("detail", e.getMessage());
            error.put("hint", "Make sure FastAPI is running: uvicorn app:app --reload");
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).body(error);
        }
    }

    /**
     * GET /api/climate/health
     * Quick check to see if both Java + Python services are alive.
     */
    @GetMapping("/health")
    public ResponseEntity<Map<String, String>> healthCheck() {
        Map<String, String> status = new HashMap<>();
        status.put("java_backend", "UP");

        try {
            ResponseEntity<Map> pythonCheck =
                    restTemplate.getForEntity("http://127.0.0.1:8000/", Map.class);
            status.put("python_ml_api", pythonCheck.getStatusCode().is2xxSuccessful()
                    ? "UP" : "DOWN");
        } catch (Exception e) {
            status.put("python_ml_api", "DOWN - " + e.getMessage());
        }

        return ResponseEntity.ok(status);
    }

    /**
     * Cascading impact analysis layer (this is YOUR logic on top of the ML output).
     * Mirrors and extends the CIS logic from ImpactForecaster.java.
     */
    private Map<String, Object> addCascadingAnalysis(
            Map<String, Object> prediction, ClimateRequest request) {

        double cropLoss      = toDouble(prediction.get("crop_loss"));
        double energyLoad    = toDouble(prediction.get("energy_load"));
        double transportDelay = toDouble(prediction.get("transport_delay"));

        // Cascading Impact Score (CIS) — same concept as ImpactForecaster
        double cis = (cropLoss + energyLoad + transportDelay) / 3.0;
        prediction.put("cascading_impact_score", Math.round(cis * 100.0) / 100.0);

        // Alert logic
        boolean alert = cis > 60 || request.getTemp() > 40;
        prediction.put("cascading_alert", alert);

        if (alert) {
            prediction.put("alert_message",
                    buildAlertMessage(cropLoss, energyLoad, transportDelay, request.getTemp()));
        } else {
            prediction.put("alert_message", "All systems normal.");
        }

        return prediction;
    }

    private String buildAlertMessage(double crop, double energy,
                                     double transport, double temp) {
        StringBuilder msg = new StringBuilder("[ALERT] ");
        if (temp > 40)    msg.append("Critical temperature. ");
        if (crop > 60)    msg.append("High crop loss — divert irrigation. ");
        if (energy > 70)  msg.append("Grid overload — reduce pump load. ");
        if (transport > 50) msg.append("Transport delays expected. ");
        return msg.toString().trim();
    }

    private double toDouble(Object val) {
        if (val == null) return 0.0;
        return Double.parseDouble(val.toString());
    }

    // -------------------------------------------------------------------------
    // Inner request DTO
    // -------------------------------------------------------------------------
    public static class ClimateRequest {
        private double temp;
        private double soilMoisture;
        private double ndvi;
        private double humidity;

        public double getTemp()         { return temp; }
        public double getSoilMoisture() { return soilMoisture; }
        public double getNdvi()         { return ndvi; }
        public double getHumidity()     { return humidity; }

        public void setTemp(double temp)                 { this.temp = temp; }
        public void setSoilMoisture(double soilMoisture) { this.soilMoisture = soilMoisture; }
        public void setNdvi(double ndvi)                 { this.ndvi = ndvi; }
        public void setHumidity(double humidity)         { this.humidity = humidity; }
    }
}