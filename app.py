import streamlit as st
import pandas as pd
import numpy as np

# Función de convolución
def apply_convolution(matrix, kernel_type='horizontal'):
    if kernel_type == 'horizontal':
        kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    elif kernel_type == 'vertical':
        kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    else:
        raise ValueError("El tipo de kernel debe ser 'horizontal' o 'vertical'")
    
    output = np.zeros_like(matrix)

    for i in range(1, matrix.shape[0] - 1):
        for j in range(1, matrix.shape[1] - 1):
            region = matrix[i-1:i+2, j-1:j+2]
            output[i, j] = np.sum(region * kernel)
    
    return output

# Función de padding
def apply_padding(matrix, padding_size):
    padded_matrix = np.pad(matrix, pad_width=padding_size, mode='constant', constant_values=0)
    return padded_matrix

# Función de stride
def apply_stride(matrix, kernel_type='horizontal', stride=2):
    if kernel_type == 'horizontal':
        kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    elif kernel_type == 'vertical':
        kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    else:
        raise ValueError("El tipo de kernel debe ser 'horizontal' o 'vertical'")

    output_rows = (matrix.shape[0] - 2) // stride + 1
    output_cols = (matrix.shape[1] - 2) // stride + 1
    output = np.zeros((output_rows, output_cols))

    row = col = 0
    for i in range(1, matrix.shape[0] - 1, stride):
        for j in range(1, matrix.shape[1] - 1, stride):
            region = matrix[i-1:i+2, j-1:j+2]
            output[row, col] = np.sum(region * kernel)
            col += 1
        col = 0
        row += 1
    
    return output

# Función de stacking
def apply_stacking(matrix, n):
    stacked_matrices = []
    for i in range(n):
        kernel_type = 'horizontal' if i % 2 == 0 else 'vertical'
        convolved = apply_convolution(matrix, kernel_type)
        stacked_matrices.append(convolved)
    
    # Retornar las matrices apiladas pero no concatenadas
    return stacked_matrices

# Función de max pooling
def apply_max_pooling(matrix, stride):
    output_rows = (matrix.shape[0] - 1) // stride + 1
    output_cols = (matrix.shape[1] - 1) // stride + 1
    pooled_matrix = np.zeros((output_rows, output_cols))

    for i in range(0, matrix.shape[0] - 1, stride):
        for j in range(0, matrix.shape[1] - 1, stride):
            region = matrix[i:i+2, j:j+2]
            pooled_matrix[i // stride, j // stride] = np.max(region)

    return pooled_matrix

# Interfaz de usuario
def main():
    st.title("Transformaciones de Imágenes")

    # Menú de opciones
    option = st.sidebar.selectbox("Seleccione la transformación", 
                                   ["Convolución", "Padding", "Stride", "Stacking", "Max Pooling"])
    
    # Cargar el archivo de píxeles
    uploaded_file = st.file_uploader("Cargar el archivo de píxeles (pixeles.xlsx)", type="xlsx")

    if uploaded_file is not None:
        # Leer la matriz de píxeles desde el archivo cargado
        pixeles_df = pd.read_excel(uploaded_file)
        pixeles_array = pixeles_df.values  # Convertir a matriz numpy
        st.write("Matriz de píxeles cargada:")
        st.write(pixeles_array)

        if option == "Convolución":
            kernel_type = st.selectbox("Selecciona el tipo de kernel", ["horizontal", "vertical"])
            if st.button("Calcular"):
                result = apply_convolution(pixeles_array, kernel_type)
                st.write("Resultado de la convolución:")
                st.write(result)

        elif option == "Padding":
            padding_size = st.number_input("Ingrese el tamaño del padding:", min_value=0, value=1)
            if st.button("Calcular"):
                result = apply_padding(pixeles_array, padding_size)
                st.write("Resultado del padding:")
                st.write(result)

        elif option == "Stride":
            kernel_type = st.selectbox("Selecciona el tipo de kernel", ["horizontal", "vertical"])
            stride = st.number_input("Ingrese el tamaño del stride:", min_value=1, value=2)
            if st.button("Calcular"):
                result = apply_stride(pixeles_array, kernel_type, stride)
                st.write("Resultado del stride:")
                st.write(result)

        elif option == "Stacking":
            n = st.number_input("Ingrese la cantidad de mapas a generar:", min_value=1, value=3)
            if st.button("Calcular"):
                result = apply_stacking(pixeles_array, n)
                st.write("Resultado del stacking:")
                # Mostrar cada matriz apilada por separado
                for i, matrix in enumerate(result):
                    st.write(f"Matriz {i+1}:")
                    st.write(matrix)

        elif option == "Max Pooling":
            stride = st.number_input("Ingrese el tamaño del stride:", min_value=1, value=2)
            if st.button("Calcular"):
                result = apply_max_pooling(pixeles_array, stride)
                st.write("Resultado del max pooling:")
                st.write(result)

if __name__ == "__main__":
    main()
