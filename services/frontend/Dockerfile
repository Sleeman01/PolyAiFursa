FROM node:20-slim

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

ARG NEXT_PUBLIC_AGENT_URL
ENV NEXT_PUBLIC_AGENT_URL=$NEXT_PUBLIC_AGENT_URL

RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]