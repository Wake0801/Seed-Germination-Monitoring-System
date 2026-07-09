# Web Backend

Thu muc nay chua backend FastAPI cho dashboard `web/`.

## Chay Backend

```text
uvicorn src.webapp.api:app --reload
```

Sau do mo:

```text
http://127.0.0.1:8000/
```

Neu port `8000` dang bi chiem, dung port khac:

```text
uvicorn src.webapp.api:app --reload --port 8001
```

Sau do mo:

```text
http://127.0.0.1:8001/
```

## Tat Backend

Neu terminal dang chay `uvicorn`, bam:

```text
Ctrl + C
```

Neu khong nho terminal nao dang chay server, tim PID tren port 8000:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen | Select-Object -ExpandProperty OwningProcess
```

Sau do tat process:

```powershell
Stop-Process -Id <PID>
```

## API Chinh

- `GET /api/health`: kiem tra backend.
- `GET /api/models`: doc `configs/model_registry.json`.
- `POST /api/predict/crop`: upload mot anh crop va chay model `.pth` bang PyTorch.
- `POST /api/predict/detect`: upload anh raw hoac frame video va chay Faster R-CNN scratch de tra ve bounding box + trang thai hat.

## Model Chinh Hien Tai

He thong web uu tien Faster R-CNN scratch cho anh raw va frame video:

```text
outputs/checkpoints/faster_rcnn_scratch/best_faster_rcnn_resnet50_fpn_scratch.pth
```

Model nay khong dung pretrained weights. File `.pth` lon hon gioi han GitHub thong thuong, nen checkpoint duoc giu local hoac luu ngoai repo; code, config va report van duoc commit binh thuong.

## `.pth` Hay ONNX?

File `.pth` chay duoc neu he thong web co Python backend dung PyTorch. Day la huong hien tai cua project.

Khong can train lai model chi de xuat ONNX. ONNX chi can khi:

- muon chay inference truc tiep trong browser,
- muon dung ONNX Runtime,
- hoac deploy sang moi truong khong dung PyTorch.

Neu can ONNX cho baseline crop classifier, dung:

```text
python src/inference/export_onnx.py --model-id custom_cnn_baseline
```
