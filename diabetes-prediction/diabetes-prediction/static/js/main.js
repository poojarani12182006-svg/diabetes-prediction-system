// ---------------------------------------------------------------------------
// Mobile nav toggle
// ---------------------------------------------------------------------------
const navToggle = document.getElementById('navToggle');
const primaryNav = document.getElementById('primaryNav');

if (navToggle && primaryNav) {
  navToggle.addEventListener('click', () => {
    const isOpen = primaryNav.classList.toggle('is-open');
    navToggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
  });
}

// ---------------------------------------------------------------------------
// Prediction form: client-side validation + loading state
// ---------------------------------------------------------------------------
const predictForm = document.getElementById('predictForm');

if (predictForm) {
  const submitBtn = document.getElementById('submitBtn');
  const loadingIndicator = document.getElementById('loadingIndicator');

  const rules = {
    Pregnancies: { min: 0, max: 20 },
    Glucose: { min: 0, max: 300 },
    BloodPressure: { min: 0, max: 200 },
    SkinThickness: { min: 0, max: 100 },
    Insulin: { min: 0, max: 900 },
    BMI: { min: 0, max: 80 },
    DiabetesPedigreeFunction: { min: 0, max: 3 },
    Age: { min: 1, max: 120 },
  };

  function validateField(input) {
    const rule = rules[input.name];
    if (!rule) return true;

    const value = parseFloat(input.value);
    const isValid = input.value.trim() !== '' && !isNaN(value) && value >= rule.min && value <= rule.max;

    input.classList.toggle('field-error', !isValid);
    return isValid;
  }

  predictForm.querySelectorAll('input[type="number"]').forEach((input) => {
    input.addEventListener('blur', () => validateField(input));
    input.addEventListener('input', () => {
      if (input.classList.contains('field-error')) validateField(input);
    });
  });

  predictForm.addEventListener('submit', (e) => {
    let allValid = true;
    predictForm.querySelectorAll('input[type="number"]').forEach((input) => {
      if (!validateField(input)) allValid = false;
    });

    if (!allValid) {
      e.preventDefault();
      return;
    }

    // Show loading state while the server processes the prediction
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.textContent = 'Analyzing…';
    }
    if (loadingIndicator) {
      loadingIndicator.classList.add('is-active');
    }
  });
}

// ---------------------------------------------------------------------------
// Result gauge — animated semi-circular confidence readout
// ---------------------------------------------------------------------------
function renderGauge() {
  const gaugeEl = document.getElementById('confidenceGauge');
  if (!gaugeEl) return;

  const confidence = parseFloat(gaugeEl.dataset.confidence || '0');
  const isDiabetic = gaugeEl.dataset.risk === 'true';
  const color = isDiabetic ? '#B8441C' : '#0F6E5C';
  const track = '#DCE5E6';

  const radius = 70;
  const circumference = Math.PI * radius; // half circle
  const offset = circumference - (confidence / 100) * circumference;

  gaugeEl.innerHTML = `
    <svg width="180" height="100" viewBox="0 0 180 100">
      <path d="M 10 90 A 80 80 0 0 1 170 90" fill="none" stroke="${track}" stroke-width="14" stroke-linecap="round"/>
      <path d="M 10 90 A 80 80 0 0 1 170 90" fill="none" stroke="${color}" stroke-width="14"
            stroke-linecap="round" stroke-dasharray="${circumference}" stroke-dashoffset="${circumference}"
            style="transition: stroke-dashoffset 900ms cubic-bezier(.22,.8,.28,1);" id="gaugeArc"/>
    </svg>
  `;

  requestAnimationFrame(() => {
    const arc = document.getElementById('gaugeArc');
    if (arc) arc.style.strokeDashoffset = offset;
  });
}

document.addEventListener('DOMContentLoaded', renderGauge);
