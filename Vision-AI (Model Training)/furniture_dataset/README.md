# Furniture Condition Dataset

## 📁 Folder Structure

```
furniture_dataset/
├── chairs/
│   ├── new/          ← Add NEW chair images here
│   ├── broken/       ← Add BROKEN chair images here
│   └── wornout/      ← Add WORNOUT chair images here
├── sofas/
│   ├── new/          ← Add NEW sofa images here
│   ├── broken/       ← Add BROKEN sofa images here
│   └── wornout/      ← Add WORNOUT sofa images here
└── tables/
    ├── new/          ← Add NEW table images here
    ├── broken/       ← Add BROKEN table images here
    └── wornout/      ← Add WORNOUT table images here
```

## 📸 How to Add Images

### For Each Furniture Type:

1. **NEW Images**: 
   - Clean, undamaged furniture
   - Good lighting
   - Clear view of the entire piece
   - Examples: Brand new chairs, freshly painted furniture

2. **BROKEN Images**:
   - Damaged furniture with visible defects
   - Cracks, chips, missing parts
   - Examples: Broken chair legs, cracked table tops, torn sofa cushions

3. **WORNOUT Images**:
   - Used furniture showing wear and tear
   - Faded colors, scratches, minor damage
   - Examples: Old office chairs, worn dining tables, faded sofas

## 📊 Recommended Amount

- **Minimum**: 50 images per condition per furniture type
- **Ideal**: 100+ images per condition per furniture type
- **Total**: 450+ images minimum (50 × 3 conditions × 3 furniture types)

## 🎯 Image Quality Tips

- Use good lighting
- Take photos from multiple angles
- Include close-ups of damage/wear
- Ensure furniture is clearly visible
- Avoid blurry or dark images

## 🔄 After Adding Images

Run the training script to improve accuracy:
```bash
python train_with_real_data.py
```

This will create a much more accurate model for furniture condition detection!
