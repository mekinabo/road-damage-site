'use strict';
(function () {
  var CLASSES = { D00: 'продольная трещина', D10: 'поперечная трещина', D20: 'сетка трещин', D40: 'выбоина' };
  var MAX_SIDE = 1600;        // фото уменьшается в браузере: быстрее загрузка с телефона, модели больше не нужно
  var KEEP_BYTES = 3 * 1048576;  // небольшие фото уходят как есть, без потери качества от пересжатия
  var TIMEOUT_MS = 90000;
  var maxMb = 20;

  var $ = function (id) { return document.getElementById(id); };
  var cv = $('cv'), ctx = cv.getContext('2d');
  var src = null, result = null, hover = -1, selected = -1, busy = false, token = 0;

  function fmt(n) { return Number(n).toLocaleString('ru-RU'); }
  function pct(conf) { return Math.round(conf * 100) + ' %'; }
  function colors() {
    var s = getComputedStyle(document.documentElement), c = {};
    ['D00', 'D10', 'D20', 'D40', 'other'].forEach(function (k) { c[k] = s.getPropertyValue('--' + k).trim(); });
    return c;
  }
  function colorOf(c, code) { return c[CLASSES[code] ? code : 'other']; }
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;   // только textContent: имена классов приходят из файла модели
    return e;
  }

  // ошибки с понятным текстом для пользователя; остальные (внутренние ошибки браузера) заменяются общим
  function userError(msg) { var e = new Error(msg); e.user = true; return e; }

  function showError(msg) {
    $('error').textContent = msg || '';
    $('error').hidden = !msg;
  }
  function setBusy(on, text) {
    busy = on;
    $('spin').hidden = !on;
    $('spintext').textContent = text || 'Анализ…';
    $('pick').disabled = $('shoot').disabled = on;
    $('drop').setAttribute('aria-busy', on ? 'true' : 'false');
  }

  // ---------- сведения о модели ----------
  function loadModelInfo(attempt) {
    fetch('/api/model', { cache: 'no-store' }).then(function (r) {
      if (!r.ok) throw new Error();
      return r.json();
    }).then(function (m) {
      maxMb = m.max_upload_mb || maxMb;
      $('maxmb').textContent = maxMb;
      $('notice').textContent = '';
      if (!m.road_model) {
        var w = el('div', 'warn');
        w.appendChild(document.createTextNode('Загружена общая модель '));
        w.appendChild(el('b', null, m.file));
        w.appendChild(document.createTextNode(', она не обучена на дорогах. Положите веса в папку models/ и перезапустите сервер.'));
        $('notice').appendChild(w);
      }
    }).catch(function () {
      $('notice').textContent = '';
      $('notice').appendChild(el('div', 'warn', 'Сервер пока не отвечает. Проверьте, что он запущен; страница попробует снова.'));
      setTimeout(function () { loadModelInfo((attempt || 0) + 1); }, Math.min(30000, 2000 * Math.pow(2, attempt || 0)));
    });
  }

  // ---------- выбор файла ----------
  function choose(input) { if (!busy) { input.value = ''; input.click(); } }
  $('pick').addEventListener('click', function () { choose($('file')); });
  $('shoot').addEventListener('click', function () { choose($('camera')); });
  $('drop').addEventListener('click', function () { if (!src) choose($('file')); });
  $('drop').addEventListener('keydown', function (e) {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); choose($('file')); }
  });
  ['file', 'camera'].forEach(function (id) {
    $(id).addEventListener('change', function (e) { if (e.target.files && e.target.files[0]) run(e.target.files[0]); });
  });

  // перетаскивание: файл, брошенный мимо зоны, не должен открываться вместо сайта
  ['dragover', 'drop'].forEach(function (t) { window.addEventListener(t, function (e) { e.preventDefault(); }); });
  ['dragover', 'dragenter'].forEach(function (t) {
    $('drop').addEventListener(t, function () { $('drop').classList.add('over'); });
  });
  ['dragleave', 'drop'].forEach(function (t) {
    $('drop').addEventListener(t, function () { $('drop').classList.remove('over'); });
  });
  $('drop').addEventListener('drop', function (e) {
    var f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) run(f);
  });
  document.addEventListener('paste', function (e) {
    var items = (e.clipboardData && e.clipboardData.files) || [];
    for (var i = 0; i < items.length; i++) {
      if (/^image\//.test(items[i].type)) { e.preventDefault(); run(items[i]); return; }
    }
  });

  $('conf').addEventListener('input', function () {
    $('confv').textContent = $('conf').value + ' %';
    selected = hover = -1;
    render();
  });
  $('labels').addEventListener('change', draw);

  // ---------- подготовка фото в браузере ----------
  function decode(file) {
    return new Promise(function (resolve) {
      var url = URL.createObjectURL(file), img = new Image();
      img.onload = function () { URL.revokeObjectURL(url); resolve(img.naturalWidth ? img : null); };
      img.onerror = function () { URL.revokeObjectURL(url); resolve(null); };
      img.src = url;   // <img> сам учитывает поворот из EXIF во всех современных браузерах
    });
  }

  function shrink(img, file) {
    var small = Math.max(img.naturalWidth, img.naturalHeight) <= MAX_SIDE && file.size <= KEEP_BYTES
      && /^image\/(jpeg|png|webp)$/.test(file.type);
    var k = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
    return new Promise(function (resolve) {
      try {   // на iPhone при нехватке памяти canvas может не создаться: тогда фото уйдёт на сервер как есть
        var c = document.createElement('canvas');
        c.width = Math.max(1, Math.round(img.naturalWidth * k));
        c.height = Math.max(1, Math.round(img.naturalHeight * k));
        var g = c.getContext('2d');
        if (!g) return resolve(null);
        g.drawImage(img, 0, 0, c.width, c.height);
        if (small) return resolve({ canvas: c, blob: file });  // сервер повернёт по EXIF так же, как <img>
        c.toBlob(function (b) { resolve(b ? { canvas: c, blob: b } : null); }, 'image/jpeg', 0.9);
      } catch (e) { resolve(null); }
    });
  }

  function looksLikeImage(file) {
    return /^image\//.test(file.type) || /\.(jpe?g|png|webp|bmp|gif|tiff?|heic|heif|avif)$/i.test(file.name || '');
  }

  // ---------- анализ ----------
  function run(file) {
    if (busy) return;
    showError('');
    if (!looksLikeImage(file)) return showError('Это не изображение. Выберите фото в формате JPG, PNG, WEBP или HEIC.');
    if (file.size > maxMb * 1048576) return showError('Файл больше ' + maxMb + ' МБ. Выберите фото поменьше.');
    if (!navigator.onLine) return showError('Нет подключения к интернету.');

    var my = ++token;
    setBusy(true, 'Подготовка фото…');
    decode(file).then(function (img) {
      return img ? shrink(img, file) : null;
    }).then(function (prepared) {
      var fd = new FormData();
      if (prepared) {
        fd.append('file', prepared.blob, prepared.blob === file ? (file.name || 'photo') : 'photo.jpg');
      } else {   // браузер не умеет показать формат (например, HEIC в Chrome): сервер пришлёт превью
        fd.append('file', file, file.name || 'photo');
        fd.append('preview', '1');
      }
      setBusy(true, 'Анализ…');
      return send(fd, 0, my)
        .then(function (data) {
          if (data.preview) return loadPreview(data.preview).then(function (im) { return { data: data, src: im }; });
          return { data: data, src: prepared.canvas };
        });
    }).then(function (out) {
      if (my !== token) return;
      if (!out.src) throw userError('Не удалось показать это фото в браузере.');
      src = out.src; result = out.data; hover = selected = -1;
      cv.width = result.width; cv.height = result.height;
      $('drop').classList.add('has-img');
      $('save').disabled = false;
      render();
    }).catch(function (e) {
      if (my === token) showError(e && e.user ? e.message : 'Что-то пошло не так. Попробуйте ещё раз или выберите другое фото.');
    }).then(function () {
      if (my === token) setBusy(false);
    });
  }

  // если сервер занят (503), ждём сколько он просит и пробуем снова, до MAX_RETRIES раз
  var MAX_RETRIES = 5;
  function send(fd, attempt, my) {
    var ctrl = window.AbortController ? new AbortController() : null;
    var timer = setTimeout(function () { if (ctrl) ctrl.abort(); }, TIMEOUT_MS);
    return fetch('/api/detect', { method: 'POST', body: fd, signal: ctrl ? ctrl.signal : undefined })
      .then(function (r) {
        clearTimeout(timer);
        return r.text().then(function (t) {
          var data = null;
          try { data = JSON.parse(t); } catch (e) { /* например, HTML-страница ошибки прокси */ }
          if (r.status === 503 && attempt < MAX_RETRIES && my === token) {
            var wait = Math.min(15, parseInt(r.headers.get('Retry-After'), 10) || 3) + Math.random() * 2;
            setBusy(true, 'Сервер занят, повторяю через ' + Math.round(wait) + ' с…');
            return new Promise(function (res) { setTimeout(res, wait * 1000); }).then(function () {
              setBusy(true, 'Анализ…');
              return send(fd, attempt + 1, my);
            });
          }
          if (!r.ok || !data) throw userError(errorText(r.status, data));
          return data;
        });
      }, function (e) {
        clearTimeout(timer);
        throw userError(e && e.name === 'AbortError'
          ? 'Сервер слишком долго не отвечает. Попробуйте ещё раз.'
          : 'Не удалось связаться с сервером. Проверьте подключение и что сервер запущен.');
      });
  }

  function errorText(status, data) {
    if (data && typeof data.detail === 'string') return data.detail;
    if (status === 413) return 'Файл слишком большой.';
    if (status === 429) return 'Слишком много запросов, подождите минуту.';
    if (status === 502 || status === 503 || status === 504) return 'Сервер занят или недоступен, попробуйте через несколько секунд.';
    return 'Ошибка сервера (' + status + '). Попробуйте ещё раз.';
  }

  function loadPreview(dataUrl) {
    return new Promise(function (resolve) {
      var im = new Image();
      im.onload = function () { resolve(im); };
      im.onerror = function () { resolve(null); };
      im.src = dataUrl;
    });
  }

  // ---------- отрисовка ----------
  function visible() {
    var t = $('conf').value / 100;
    if (!result) return [];
    return result.detections.map(function (d, i) {
      return { code: d.code, name: d.name, conf: d.conf, box: d.box, share: d.frame_share, i: i };
    }).filter(function (d) { return d.conf >= t - 1e-9; });
  }

  function render() {
    if (!src) return;
    draw();
    renderList();
  }

  function draw() {
    if (!src) return;
    var c = colors(), W = cv.width, H = cv.height;
    ctx.drawImage(src, 0, 0, W, H);
    // размеры в пикселях снимка, но не мельче 2 px линии и 12 px текста на экране (важно для телефонов)
    var k = W / Math.max(1, cv.clientWidth || W);
    var ds = visible(), lw = Math.max(2 * k, W / 400), fs = Math.round(Math.max(12 * k, W / 55));
    ctx.font = '600 ' + fs + 'px system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif';
    ctx.textBaseline = 'top';
    var focus = selected >= 0 ? selected : hover;
    // выделенную рамку рисуем последней, чтобы её не перекрывали
    ds.sort(function (a, b) { return (a.i === focus) - (b.i === focus); });
    var placed = [];  // уже нарисованные подписи, чтобы новые их не закрывали
    ds.forEach(function (d) {
      var x1 = d.box[0], y1 = d.box[1], x2 = d.box[2], y2 = d.box[3], col = colorOf(c, d.code);
      ctx.globalAlpha = focus >= 0 && d.i !== focus ? 0.45 : 1;
      ctx.lineWidth = d.i === focus ? lw * 2.5 : lw;
      ctx.strokeStyle = col;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      if ($('labels').checked) {
        var txt = d.name + ' ' + pct(d.conf), w = ctx.measureText(txt).width + 10, h = fs + 8;
        var tx = Math.max(0, Math.min(x1, W - w)), ty = y1 >= h ? y1 - h : Math.min(y1, H - h);
        for (var n = 0; n < 12; n++) {
          var hit = placed.some(function (r) { return tx < r[0] + r[2] && r[0] < tx + w && ty < r[1] + r[3] && r[1] < ty + h; });
          if (!hit || ty + 2 * h > H) break;
          ty += h;
        }
        placed.push([tx, ty, w, h]);
        ctx.fillStyle = col; ctx.fillRect(tx, ty, w, h);
        ctx.fillStyle = '#fff'; ctx.fillText(txt, tx + 5, ty + 4);
      }
    });
    ctx.globalAlpha = 1;
  }

  function renderList() {
    var c = colors(), ds = visible(), counts = {};
    ds.forEach(function (d) { counts[d.code] = (counts[d.code] || 0) + 1; });
    var codes = Object.keys(CLASSES);
    Object.keys(counts).forEach(function (k) { if (codes.indexOf(k) < 0) codes.push(k); });

    var sum = $('summary');
    sum.textContent = '';
    codes.forEach(function (code) {
      var s = el('div', 'stat'), t = el('div', 't'), dot = el('span', 'dot');
      dot.style.background = colorOf(c, code);
      s.appendChild(el('div', 'n', String(counts[code] || 0)));
      t.appendChild(dot); t.appendChild(document.createTextNode(CLASSES[code] || code));
      s.appendChild(t); sum.appendChild(s);
    });

    var list = $('list');
    list.textContent = '';
    if (!ds.length) {
      list.appendChild(el('li', 'muted', 'Предполагаемых повреждений с уверенностью от ' + $('conf').value + ' % не найдено. Попробуйте уменьшить порог.'));
    }
    ds.forEach(function (d) {
      var li = el('li'), b = el('button'), left = el('span'), dot = el('span', 'dot');
      b.type = 'button';
      b.setAttribute('aria-pressed', d.i === selected ? 'true' : 'false');
      if (d.i === selected) b.className = 'sel';
      dot.style.background = colorOf(c, d.code);
      left.appendChild(dot); left.appendChild(document.createTextNode(d.name));
      left.appendChild(el('small', null, 'рамка — ' + fmt(d.share) + ' % кадра'));
      b.appendChild(left); b.appendChild(el('b', null, pct(d.conf)));
      // подсветка при наведении только для мыши: на телефоне касание даёт «наведение», которое потом не снимается
      var isMouse = function (e) { return !e.pointerType || e.pointerType === 'mouse'; };
      var enter = window.PointerEvent ? 'pointerenter' : 'mouseenter', leave = window.PointerEvent ? 'pointerleave' : 'mouseleave';
      b.addEventListener(enter, function (e) { if (isMouse(e)) { hover = d.i; draw(); } });
      b.addEventListener(leave, function (e) { if (isMouse(e)) { hover = -1; draw(); } });
      b.addEventListener('click', function () { hover = -1; selected = selected === d.i ? -1 : d.i; render(); });
      li.appendChild(b); list.appendChild(li);
    });

    var names = { 'm3_web.pt': 'M3 (дообученная)', 'm1_finetuned.pt': 'M1 (дообученная)', 'm0_rdd2022.pt': 'M0 (RDD2022)' };
    $('meta').textContent = 'Модель: ' + (names[result.model] || result.model) + ' · время анализа ' + fmt(result.ms) + ' мс · размер '
      + result.width + '×' + result.height;
  }

  // ---------- сохранение ----------
  $('save').addEventListener('click', function () {
    if (!src) return;
    cv.toBlob(function (b) {
      if (!b) return showError('Не удалось сохранить картинку.');
      var a = document.createElement('a'), url = URL.createObjectURL(b);
      a.href = url; a.download = 'road-damage-result.jpg';
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(function () { URL.revokeObjectURL(url); }, 10000);
    }, 'image/jpeg', 0.92);
  });

  if (window.matchMedia) {   // смена светлой/тёмной темы
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    if (mq.addEventListener) mq.addEventListener('change', render);
  }
  var resizeTimer;
  window.addEventListener('resize', function () { clearTimeout(resizeTimer); resizeTimer = setTimeout(draw, 150); });
  window.addEventListener('offline', function () { showError('Нет подключения к интернету.'); });
  window.addEventListener('online', function () { showError(''); });

  loadModelInfo(0);
})();
