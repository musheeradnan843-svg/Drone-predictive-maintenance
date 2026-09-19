const API_URL="http://127.0.0.1:8000/predict";
const form=document.getElementById("predictionForm");
const btn=document.getElementById("predictBtn");
const empty=document.getElementById("empty");
const results=document.getElementById("results");
const badge=document.getElementById("badge");
const error=document.getElementById("error");

form.addEventListener("submit",async(e)=>{
  e.preventDefault();
  const data={
    Type:document.getElementById("Type").value,
    Air_Temperature:Number(document.getElementById("Air_Temperature").value),
    Process_Temperature:Number(document.getElementById("Process_Temperature").value),
    Rotational_Speed:Number(document.getElementById("Rotational_Speed").value),
    Torque:Number(document.getElementById("Torque").value),
    Tool_Wear:Number(document.getElementById("Tool_Wear").value)
  };
  btn.disabled=true; btn.innerHTML="ANALYZING SENSOR DATA...  ⋯"; error.classList.add("hidden");
  try{
    const response=await fetch(API_URL,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)});
    if(!response.ok) throw new Error(`API returned ${response.status}`);
    showResults(await response.json());
  }catch(err){
    empty.classList.add("hidden"); results.classList.remove("hidden");
    error.textContent="FastAPI se connection nahi ho raha. Uvicorn running hai aur CORS enabled hai, ye check karo.";
    error.classList.remove("hidden"); badge.textContent="CONNECTION ERROR"; badge.className="alert";
  }finally{
    btn.disabled=false; btn.innerHTML="◈ &nbsp; RUN MAINTENANCE SCAN <b>→</b>";
  }
});

function showResults(r){
  empty.classList.add("hidden"); results.classList.remove("hidden");
  const failure=Number(r.failure_prediction), anomaly=Number(r.anomaly_prediction), score=Number(r.anomaly_score);
  document.getElementById("failureStatus").textContent=r.failure_status;
  document.getElementById("failurePrediction").textContent=failure;
  document.getElementById("anomalyStatus").textContent=r.anomaly_status;
  document.getElementById("anomalyPrediction").textContent=anomaly;
  document.getElementById("anomalyScore").textContent=score.toFixed(4);
  document.getElementById("failureCard").classList.toggle("alert",failure===1);
  document.getElementById("anomalyCard").classList.toggle("alert",anomaly===-1);
  const visual=Math.max(0,Math.min(100,(score+0.5)*100));
  document.getElementById("scoreFill").style.width=visual+"%";
  if(failure===1||anomaly===-1){badge.textContent="ATTENTION";badge.className="alert"}
  else{badge.textContent="SCAN CLEAR";badge.className=""}
}
