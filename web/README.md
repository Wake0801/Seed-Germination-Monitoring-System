# Seed Germination Monitoring Web

Thu muc nay chua giao dien dashboard cho he thong giam sat nay mam hat giong.

## Cau Truc

```text
web/
+-- index.html
+-- assets/
    +-- css/dashboard.css
    +-- js/data.js
    +-- js/api.js
    +-- js/ui.js
    +-- js/app.js
```

`index.html` giu layout chinh. CSS, du lieu mo phong, API adapter va logic UI duoc tach rieng de de bao tri.

## Chay Web Voi Backend PyTorch

Day la cach chay dung neu muon dung checkpoint `.pth` that:

```text
uvicorn src.webapp.api:app --reload
```

Sau do mo:

```text
http://127.0.0.1:8000/
```

Backend cung cap:

- `GET /api/models`
- `POST /api/predict/crop`
- `POST /api/predict/detect`

Monitoring System hien chi cho chon model object detection Faster R-CNN. Baseline Custom CNN van duoc giu trong tab Model Comparison de so sanh ket qua, khong dung cho demo anh raw.

## Demo Anh Va Video

Anh demo co san tai:

```text
data/demo/raw_images/
```

Video demo co san tai:

```text
data/demo/videos/pg1_1_timelapse.mp4
```

Cach demo:

1. Chay backend bang `uvicorn src.webapp.api:app --reload`.
2. Mo `http://127.0.0.1:8000/`.
3. Bam `Choose File`.
4. Chon mot anh trong `data/demo/raw_images/`, hoac chon video `data/demo/videos/pg1_1_timelapse.mp4`.
5. Bam `Analyze`.

Neu upload anh, web goi Faster R-CNN mot lan va ve bounding box len anh.

Neu upload video, web trich toi da 8 frame mau, gui tung frame vao Faster R-CNN, sau do hien thanh frame slider de xem ket qua theo tung frame mau. Day la demo video o muc he thong, khong can train lai model.

## Tat Web

Web khong tu chay vinh vien. No chi con mo khi server/terminal van dang chay.

Neu dang chay `uvicorn` trong terminal, bam:

```text
Ctrl + C
```

Neu chi mo truc tiep file `web/index.html`, chi can dong tab trinh duyet vi khong co backend server.

Neu quen server dang chay o terminal nao, kiem tra process dang nghe port 8000:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen | Select-Object -ExpandProperty OwningProcess
```

Sau khi biet PID, tat process do:

```powershell
Stop-Process -Id <PID>
```

## `.pth` Hay ONNX?

File `.pth` chay duoc voi backend Python/PyTorch hien tai. Khong can export ONNX tru khi muon chay inference truc tiep trong browser, dung ONNX Runtime, hoac deploy sang moi truong khong cai PyTorch.
