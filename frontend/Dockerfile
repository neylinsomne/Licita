FROM node:18-alpine

WORKDIR /app

COPY package.json .

# Install dependencies
RUN npm install

# Copy source
COPY . .

# Expose Vite port
EXPOSE 3000

# Run in dev mode
CMD ["npm", "run", "dev", "--", "--host"]
