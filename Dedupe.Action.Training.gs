/** Action handlers split for safer per-area changes */

function donor_createTrainingPairs() {
  const rows = donor_loadAllRows_();
  const blocks = donor_buildBlocks_(rows);
  const pairs = donor_generatePairsFromBlocks_(blocks);
  const sampled = donor_sampleArray_(pairs, DONOR_CFG.sampleTrainingPairs);
  donor_writeTrainingSheet_(sampled);
  SpreadsheetApp.getActive().toast('Training sheet ready. Label 1 or 0, then run Train matcher.', 'Donor Matcher', 8);
}
function donor_trainMatcher() {
  const sh = donor_getOrCreateSheet_(DONOR_CFG.trainingSheet);
  const vals = sh.getDataRange().getValues();
  if (vals.length < 2) {
    SpreadsheetApp.getUi().alert('No training data found. Run Create training pairs.');
    return;
  }
  const h = vals[0].map(s => String(s || '').trim());
  function idx(name) {
    const i = h.indexOf(name);
    if (i === -1) throw new Error('Missing column: ' + name);
    return i;
  }
  const featureCols = [
    'feat_name','feat_last','feat_addr','feat_emp','feat_occ','feat_zip','feat_city',
    'feat_name_last','feat_name_addr','feat_name_emp','feat_name_occ',
    'feat_last_addr','feat_last_emp','feat_last_occ',
    'feat_addr_emp','feat_addr_occ','feat_emp_occ'
  ];
  const I = featureCols.map(idx);
  const iLabel = idx('Label');

  const X = [];
  const y = [];
  for (let r = 1; r < vals.length; r++) {
    const row = vals[r];
    const label = Number(row[iLabel]);
    if (label !== 0 && label !== 1) continue;
    const feat = [];
    for (let k = 0; k < I.length; k++) feat.push(Number(row[I[k]] || 0));
    X.push(feat);
    y.push(label);
  }
  if (!X.length) {
    SpreadsheetApp.getUi().alert('No labeled rows. Set Label to 1 or 0.');
    return;
  }
  const w = donor_fitLogReg_(X, y, DONOR_CFG.learningRate, DONOR_CFG.maxTrainIterations);
  PropertiesService.getDocumentProperties().setProperty('donor_model_weights', JSON.stringify(w));
  donor_buildAndStoreFirstNamePairTable_();
  SpreadsheetApp.getUi().alert('Training complete. Weights and first name table saved.');
}
function donor_assignDonorIds() {
  const wRaw = PropertiesService.getDocumentProperties().getProperty('donor_model_weights');
  if (!wRaw) {
    SpreadsheetApp.getUi().alert('No trained model found. Run Train matcher first.');
    return;
  }
  const w = JSON.parse(wRaw);
  const rows = donor_loadAllRows_();
  const blocks = donor_buildBlocks_(rows);
  const pairs = donor_generatePairsFromBlocks_(blocks);
  const uf = donor_newUnionFind_(rows.length);
  let kept = 0;
  for (let i = 0; i < pairs.length; i++) {
    const f = pairs[i].features;
    const x = [
      f.nameSim, f.lastSim, f.addrSim, f.empSim, f.occSim, f.zipMatch, f.cityMatch,
      f.name_last, f.name_addr, f.name_emp, f.name_occ,
      f.last_addr, f.last_emp, f.last_occ,
      f.addr_emp, f.addr_occ, f.emp_occ
    ];
    const prob = donor_predictLogReg_(x, w);
    if (prob >= DONOR_CFG.predictThreshold) {
      donor_union_(uf, pairs[i].aIdx, pairs[i].bIdx);
      kept++;
    }
  }
  const comps = donor_components_(uf);
  const idMap = new Map();
  let nextId = 1;
  for (let c = 0; c < comps.length; c++) {
    const comp = comps[c];
    for (let j = 0; j < comp.length; j++) idMap.set(comp[j], nextId);
    nextId++;
  }
  donor_writeIdsToSheets_(rows, idMap);
  SpreadsheetApp.getActive().toast('Assigned DonorIDs. Matches kept: ' + kept, 'Donor Matcher', 8);
}
function donor_addUncertainPairs() {
  const weightsRaw = PropertiesService.getDocumentProperties().getProperty('donor_model_weights');
  const rows = donor_loadAllRows_();
  const blocks = donor_buildBlocks_(rows);
  const pairs = donor_generatePairsFromBlocks_(blocks);

  if (!weightsRaw) {
    const seed = donor_sampleArray_(pairs, DONOR_CFG.initialSamplePairs);
    donor_appendPairsToTraining_(seed);
    SpreadsheetApp.getUi().alert('Added ' + seed.length + ' seed pairs. Label them, train, then run again.');
    return;
  }
  const w = JSON.parse(weightsRaw);
  const seen = donor_getSeenPairKeys_();
  const band = DONOR_CFG.uncertaintyBand;
  const lower = 0.5 - band;
  const upper = 0.5 + band;
  const candidates = [];
  for (let i = 0; i < pairs.length; i++) {
    const p = pairs[i];
    const key = donor_pairKey_(p.a, p.b);
    if (seen.has(key)) continue;
    const f = p.features;
    const x = [
      f.nameSim, f.lastSim, f.addrSim, f.empSim, f.occSim, f.zipMatch, f.cityMatch,
      f.name_last, f.name_addr, f.name_emp, f.name_occ,
      f.last_addr, f.last_emp, f.last_occ,
      f.addr_emp, f.addr_occ, f.emp_occ
    ];
    const prob = donor_predictLogReg_(x, w);
    if (prob >= lower && prob <= upper) {
      candidates.push({ pair: p, uncertainty: Math.abs(prob - 0.5) });
    }
  }
  candidates.sort((a, b) => a.uncertainty - b.uncertainty);
  const take = Math.min(DONOR_CFG.uncertainBatchSize, candidates.length);
  const toAppend = [];
  for (let j = 0; j < take; j++) toAppend.push(candidates[j].pair);
  if (!toAppend.length) {
    SpreadsheetApp.getUi().alert('No uncertain pairs in current band. Increase uncertaintyBand or add data.');
    return;
  }
  donor_appendPairsToTraining_(toAppend);
  SpreadsheetApp.getUi().alert('Added ' + toAppend.length + ' uncertain pairs. Label them and retrain.');
}
function dl_prepareIncrementalTrainingJob_() {
  dl_resetProgress_();
  dl_setProgress_('Preparing incremental training job', 0, 1, 'Creating Drive bundle');

  const webAppUrl = dl_getExecUrlFromOptions_();
  if (!webAppUrl) {
    SpreadsheetApp.getUi().alert(
      'Put your Web App URL ending with /exec in Options!I2.\nExample:\nhttps://script.google.com/macros/s/AKfycb.../exec'
    );
    return;
  }

  const inputCfg = dl_getInputSheetConfig_();
  const kref = dl_exportSheetAsCsv_(inputCfg.krefSheetName);
  const fec  = dl_exportSheetAsCsv_(inputCfg.fecSheetName);
  if (!kref.file || !fec.file) {
    SpreadsheetApp.getUi().alert(
      'Missing input sheets or no rows.\n' +
      'Configured KREF sheet: ' + inputCfg.krefSheetName + ' (' + inputCfg.krefRows + ' rows)\n' +
      'Configured FEC sheet: ' + inputCfg.fecSheetName + ' (' + inputCfg.fecRows + ' rows)\n\n' +
      'Optional override: set Options!K2 (KREF sheet) and Options!L2 (FEC sheet).'
    );
    return;
  }

  // Share temporary CSV files for direct download URLs used by the runner
  kref.file.setSharing(DriveApp.Access.ANYONE_WITH_LINK, DriveApp.Permission.VIEW);
  fec.file.setSharing(DriveApp.Access.ANYONE_WITH_LINK, DriveApp.Permission.VIEW);

  const krefUrl = 'https://drive.google.com/uc?export=download&id=' + kref.file.getId();
  const fecUrl = 'https://drive.google.com/uc?export=download&id=' + fec.file.getId();

  const token = dl_makeToken_();
  const until = Date.now() + DL_CFG.tokenMinutes * 60 * 1000;

  // Read match threshold from Options!J2
  const ss = SpreadsheetApp.getActive();
  const optionsSheet = ss.getSheetByName('Options');
  const threshold = optionsSheet ? Number(optionsSheet.getRange('J2').getValue() || 0.7) : 0.7;

  const job = {
    jobId: 'job_' + Utilities.getUuid().replace(/-/g, '').slice(0, 12),
    mode: 'incremental',  // NEW: Flag to indicate incremental training mode
    cfg: {
      sampleTrainingPairs: DL_CFG.sampleTrainingPairs,
      uncertainBatchSize: DL_CFG.uncertainBatchSize,
      uncertaintyBand: DL_CFG.uncertaintyBand,
      maxPairsPerBlock: DL_CFG.maxPairsPerBlock,
      maxTotalPairs: DL_CFG.maxTotalPairs,
      predictThreshold: threshold
    },
    krefUrl: krefUrl,
    fecUrl:  fecUrl,
    modelUrl: webAppUrl + '?model=1',
    resultUrl: webAppUrl + '?result=1&token=' + encodeURIComponent(token)
  };

  const props = PropertiesService.getDocumentProperties();
  props.setProperty('dl_token', token);
  props.setProperty('dl_token_until', String(until));
  props.setProperty('dl_job_json', JSON.stringify(job));
  props.setProperty('dl_kref_fileId', kref.file.getId());
  props.setProperty('dl_fec_fileId',  fec.file.getId());
  dl_setTokenCsvFileIds_(token, kref.file.getId(), fec.file.getId());
  props.setProperty('dl_staging_folderId', kref.folder.getId());

  const runnerVersion = String(Date.now());
  const expectedSpreadsheetId = SpreadsheetApp.getActive().getId();

  const cmd =
    "echo '=== Web app health endpoint ===' && " +
    "curl -sSL '" + webAppUrl + "?health=1&v=" + runnerVersion + "' -o /tmp/donor_health.json && " +
    "cat /tmp/donor_health.json && echo && " +
    "grep -q '\"ok\":true' /tmp/donor_health.json || { " +
      "echo 'ERROR: web app health endpoint did not return Apps Script JSON.'; " +
      "echo 'If you see HTML, this /exec URL is not your active Web App deployment.'; " +
      "exit 1; " +
    "} && " +
    "grep -q 'DM_LOCAL_RUNNER_20260222' /tmp/donor_health.json || { " +
      "echo 'ERROR: health fingerprint mismatch (old deployment or wrong URL).'; " +
      "exit 1; " +
    "} && " +
    "grep -q '\"spreadsheetId\":\"" + expectedSpreadsheetId + "\"' /tmp/donor_health.json || { " +
      "echo 'ERROR: Web App points to a different spreadsheet ID.'; " +
      "echo 'Expected spreadsheet ID: " + expectedSpreadsheetId + "'; " +
      "echo 'Health endpoint response:'; cat /tmp/donor_health.json; " +
      "echo 'Fix: In THIS sheet, set Options!I2 to the /exec URL deployed from this copied sheet script project.'; " +
      "exit 1; " +
    "} && " +
    "curl -sSL '" + webAppUrl + "?runner=1&v=" + runnerVersion + "' -o /tmp/donor_runner.py && " +
    "echo '=== Runner header (first 5 lines) ===' && head -n 5 /tmp/donor_runner.py && " +
    "echo '=== Runner fingerprint lines ===' && (grep -n 'RUNNER_FINGERPRINT' /tmp/donor_runner.py || echo 'No fingerprint found in fetched runner') && " +
    "grep -q 'RUNNER_FINGERPRINT: DM_LOCAL_RUNNER_20260222' /tmp/donor_runner.py || { " +
      "echo 'ERROR: fetched runner is not the expected deployment (fingerprint missing/mismatched).'; " +
      "echo 'Fix: redeploy web app and ensure Options!I2 points to that /exec URL.'; " +
      "exit 1; " +
    "} && " +
    "python3 -m py_compile /tmp/donor_runner.py || { " +
      "echo '=== RUNNER PYTHON SYNTAX CHECK FAILED ==='; " +
      "echo 'Showing lines 2918-2928 for diagnosis:'; " +
      "nl -ba /tmp/donor_runner.py | sed -n '2918,2928p'; " +
      "echo 'First suspicious print(\" occurrence:'; " +
      "grep -n 'print(\"' /tmp/donor_runner.py | head -n 1; " +
      "echo 'Tip: if fingerprint is missing above, you are hitting an old deployment URL/version.'; " +
      "exit 1; " +
    "} && " +
    "echo '=== Job bundle preflight ===' && " +
    "curl -sSL '" + webAppUrl + "?job=1&token=" + token + "' -o /tmp/donor_job.json && " +
    "grep -q '\"jobId\"' /tmp/donor_job.json || { " +
      "echo 'ERROR: job bundle endpoint did not return expected JSON (token invalid/expired).'; " +
      "echo 'Job endpoint response:'; cat /tmp/donor_job.json; " +
      "exit 1; " +
    "} && " +
    "python3 /tmp/donor_runner.py --bundle '" + webAppUrl + "?job=1&token=" + token + "' " +
    "--result '" + webAppUrl + "?result=1&token=" + token + "'";

  dl_setProgress_('Waiting', 0, 0, 'Copy the command into Terminal');

  const html = HtmlService.createHtmlOutput(
    '<div style="font-family:system-ui,Arial;padding:12px;max-width:720px">' +
      '<div style="margin:6px 0"><strong>Incremental Training Mode</strong></div>' +
      '<div style="margin:6px 0;color:#666">This will load your existing training data and let you add more labeled pairs.</div>' +
      '<div style="margin:6px 0">Token expires in about ' + DL_CFG.tokenMinutes + ' minutes</div>' +
      '<div style="margin:6px 0;font-size:11px;color:#666">Expected spreadsheet ID: <code>' + dl_htmlEscape_(expectedSpreadsheetId) + '</code></div>' +
      '<div style="margin:6px 0;font-size:12px;color:#444">Using KREF: <b>' + dl_htmlEscape_(inputCfg.krefSheetName) + '</b> (' + inputCfg.krefRows + ' rows), FEC: <b>' + dl_htmlEscape_(inputCfg.fecSheetName) + '</b> (' + inputCfg.fecRows + ' rows)</div>' +
      '<div style="margin:6px 0">Copy this command into your Mac Terminal:</div>' +
      '<textarea id="cmd" style="width:100%;height:140px" readonly>' +
        dl_htmlEscape_(cmd) +
      '</textarea>' +
      '<div style="margin-top:8px">' +
        '<button onclick="navigator.clipboard.writeText(document.getElementById(\'cmd\').value)">Copy to clipboard</button>' +
      '</div>' +
      '<div style="margin-top:8px;color:#666;font-size:12px">Keep this sheet open. Progress will appear in the sidebar.</div>' +
    '</div>'
  ).setWidth(740).setHeight(320);
  SpreadsheetApp.getUi().showModalDialog(html, 'Continue Training Locally');
}
