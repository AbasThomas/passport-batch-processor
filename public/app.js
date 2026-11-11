let selected=[],current=localStorage.getItem('passport-current-batch')||'',filter='all';const $=s=>document.querySelector(s);const api=(u,o={})=>{const request={...o};if(request.method==='POST'&&!request.body)request.body='{}';const headers=request.body?{'content-type':'application/json'}:{};return fetch(u,{...request,headers}).then(async r=>{if(!r.ok)throw Error((await r.json().catch(()=>({}))).error||r.statusText);return r.json()})};
console.log('app.js loaded, current:', current);
window.onerror=(msg,src,line,col)=>{console.error('Error:',msg,'at',src+':'+line+':'+col)};
function settings(){const [width,height]=$('#dimensions').value.split('x').map(Number);return{background:$('#background').value,preset:'natural',width,height,crop:true,format:$('#format').value,quality:Number($('#quality').value),dpi:Number($('#dpi').value),autoDetect:$('#autoDetect').checked,enhance:$('#enhance').checked}}
function choose(files){selected=[...files];$('#uploadStatus').textContent=`${selected.length} file(s) selected. Files upload one at a time, keeping browser memory bounded.`}
$('#pick').onchange=e=>choose(e.target.files);$('#folder').onchange=e=>choose(e.target.files);$('#drop').ondragover=e=>{e.preventDefault();e.currentTarget.style.background='#edf9f7'};$('#drop').ondragleave=e=>e.currentTarget.style.background='';$('#drop').ondrop=e=>{e.preventDefault();e.currentTarget.style.background='';choose(e.dataTransfer.files)};
$('#new').onclick=()=>{console.log('New batch clicked');current='';localStorage.removeItem('passport-current-batch');selected=[];$('#name').value='';$('#uploadStatus').textContent='';$('#setup').hidden=false;$('#dashboard').hidden=true;window.last=null;$('#batchName').textContent='';$('#headline').textContent='';$('#counts').innerHTML='';$('#rows').innerHTML='';$('#progress').style.width='0%';$('#export').href='#'};
$('#zip').onclick=()=>$('#zipfile').click();$('#zipfile').onchange=e=>{selected=[...e.target.files];$('#uploadStatus').textContent=`ZIP selected: ${selected[0]?.name}`};
function xhrUpload(url,file){return new Promise((resolve,reject)=>{const x=new XMLHttpRequest(),form=new FormData();form.append('file',file);x.open('POST',url);x.upload.onprogress=e=>{if(e.lengthComputable)$('#uploadStatus').textContent=`Uploading ${file.name}: ${Math.round(e.loaded/e.total*100)}%`};x.onload=()=>x.status<300?resolve(JSON.parse(x.responseText)):reject(Error(x.responseText));x.onerror=()=>reject(Error('Network error'));x.send(form)})}
$('#create').onclick=async()=>{if(!selected.length)return alert('Select photographs, a folder, or a ZIP first.');try{const b=await api('/api/batches',{method:'POST',body:JSON.stringify({name:$('#name').value,settings:settings()})});current=b.id;localStorage.setItem('passport-current-batch',current);for(let i=0;i<selected.length;i++){const f=selected[i];$('#uploadStatus').textContent=`Uploading ${i+1}/${selected.length}: ${f.name}`;if(f.name.toLowerCase().endsWith('.zip'))await xhrUpload(`/api/batches/${current}/upload-zip`,f);else await xhrUpload(`/api/batches/${current}/upload`,f)}$('#setup').hidden=true;$('#dashboard').hidden=false;await refresh()}catch(e){alert(e.message)}};
$('#start').onclick=async()=>{await api(`/api/batches/${current}/start`,{method:'POST'});refresh()};
$('#filters').onclick=e=>{if(e.target.dataset.f){filter=e.target.dataset.f;render(window.last)}};
async function refresh(){if(!current)return;try{window.last=await api(`/api/batches/${current}`);render(window.last)}catch(e){console.warn(e)}setTimeout(refresh,2000)}
function render(b){if(!b)return;$('#batchName').textContent=b.name;$('#headline').textContent=`${b.total} photographs`;$('#counts').innerHTML=Object.entries(b.counts).filter(([,n])=>n).map(([k,n])=>`<span>${n} ${k}</span>`).join('');$('#progress').style.width=b.progress+'%';$('#export').href=`/api/batches/${b.id}/export`;const fs=filter==='all'?b.files:b.files.filter(x=>x.state===filter);$('#rows').innerHTML=fs.map(x=>`<tr><td>${esc(x.sourceName)}</td><td>${x.state==='completed'?`<a href="/api/files/${x.id}/download">${esc(x.outputName)}</a> <button class="compare-btn" data-id="${x.id}">Compare</button>`:esc(x.outputName)}</td><td class="pill ${x.state}">${x.state}</td><td>${esc(x.error||'—')}${x.customSettings?' · custom settings':''}</td><td>${x.state==='failed'?`<button onclick="retry('${x.id}')">Retry</button>`:''}${['uploaded','queued'].includes(x.state)?` <button onclick="removeFile('${x.id}')">Remove</button>`:''}</td></tr>`).join('')||'<tr><td colspan="5">No files in this filter.</td></tr>';bindCompareButtons()}
function esc(s){return String(s).replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]))}window.retry=async id=>{await api(`/api/files/${id}/retry`,{method:'POST'});refresh()};window.removeFile=async id=>{await api(`/api/files/${id}`,{method:'DELETE'});refresh()};

function bindCompareButtons(){document.querySelectorAll('.compare-btn').forEach(btn=>{btn.onclick=()=>openCompare(btn.dataset.id)})}

function openCompare(fileId){
    const modal=$('#compare-modal');
    const beforeImg=$('#before-img');
    const afterImg=$('#after-img');
    const slider=$('#compare-slider');
    beforeImg.src=`/api/files/${fileId}/original`;
    afterImg.src=`/api/files/${fileId}/download`;
    modal.hidden=false;
    slider.style.left='50%';
    // slider drag
    let isDragging=false;
    const container=$('.compare-container');
    const updateSlider=(clientX)=>{
        const rect=container.getBoundingClientRect();
        const x=Math.max(0,Math.min(clientX-rect.left,rect.width));
        const pct=(x/rect.width)*100;
        slider.style.left=pct+'%';
    };
    slider.onmousedown=(e)=>{isDragging=true;e.preventDefault()};
    document.onmousemove=(e)=>{if(isDragging)updateSlider(e.clientX)};
    document.onmouseup=()=>{isDragging=false};
    container.onmousemove=(e)=>{if(!isDragging)updateSlider(e.clientX)};
    slider.onmousemove=(e)=>{if(isDragging)updateSlider(e.clientX)};
}

$('#close-compare').onclick=()=>{$('#compare-modal').hidden=true};
$('.modal-backdrop').onclick=()=>{$('#compare-modal').hidden=true};
document.addEventListener('keydown',(e)=>{if(e.key==='Escape')$('#compare-modal').hidden=true});
if(current){$('#setup').hidden=true;$('#dashboard').hidden=false;refresh()}
